"""方案 B2: 反弹感知的无真值关联 —— 在方案B基础上加入"撞墙反弹折叠".

方案B 的问题 (运动模式分析结论):
  - 反弹极其普遍: 观测段 88.2% / 未来段 91.2% 的序列发生撞墙反弹.
  - 60.7% 的数字从末帧直线外推一步就越过边界 [0,36].
  - 方案B 的 ref=q+v 是"直线外推", 在即将反弹/刚反弹时 ref 偏离真实轨迹 -> 翻转误判.

方案B2 核心:
  - 末帧速度 v_last = q_last - q_prev (已验证比整段拟合更准, 整段拟合被反弹污染).
  - 检测"末帧是否刚反弹": 若观测末两帧速度方向突变, v_last 已是反弹后速度.
  - 反弹折叠: 从 q_last 沿 v_last 外推 K 步, 每步若越过边界 [0,LIMIT] 则按反弹规则折回,
    得到"反弹感知"的参考轨迹 ref_k (k=0..K-1).
  - 用前 K 帧的平均交叉距离判断交换 (保留方案B稳健判断), 但参考用反弹折叠后的 ref_k.

无真值、纯几何、C++ 可移植. 产出: 终端报告 + results/figs/assoc_planB2.png + results/关联分析_方案B2.md

运行: E:\\anaconda3\\envs\\CurveSegmentationMethod\\python.exe assoc_planB2.py
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moving_mnist import MovingMNIST
from train_convlstm_v2 import MultiTaskConvLSTMv2, sort_slots, N_INPUT, N_FUTURE, CANVAS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_ROOT = r"D:\MovingMNIST\dataset\data"
CKPT_PATH = r"D:\MovingMNIST\checkpoints\convlstm_v4ft_best.pth"
OUT_DIR = r"D:\MovingMNIST\results"
FIG_DIR = os.path.join(OUT_DIR, "figs")
TEST_SEED = 2026
N_EVAL = 2000
K_FRAMES = 5          # 用前 K 帧平均判断交换
SWAP_FRAME_TH = 3
LIMIT = CANVAS - 28   # 活动范围 [0, LIMIT] (左上角坐标)


def load_model(device):
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    cfg = ckpt["config"]
    model = MultiTaskConvLSTMv2(cls_mode=cfg["cls_mode"], traj_mode=cfg["traj_mode"],
                                detach_anchor=cfg["detach_anchor"], h1_ch=cfg["h1_ch"],
                                h2_ch=cfg["h2_ch"], dec_hidden=cfg["dec_hidden"],
                                slot_feat=cfg.get("slot_feat", False)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def run_inference(model, loader, device):
    preds, targets, obs = [], [], []
    for batch in loader:
        video = batch["video"].to(device)
        centers = batch["centers"].to(device)
        _, centers = sort_slots(batch["labels"].to(device), centers)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits, traj, _ = model(video[:, :N_INPUT])
        preds.append((traj[:, 1:].float() * CANVAS).cpu().numpy())
        targets.append(centers[:, N_INPUT:].cpu().numpy())
        obs.append(centers[:, :N_INPUT].cpu().numpy())
    return (np.concatenate(preds, 0), np.concatenate(targets, 0),
            np.concatenate(obs, 0))


def per_seq_error(pred, target):
    err = np.linalg.norm(pred - target, axis=-1)
    return err.mean(axis=(1, 2)), err[:, -1].mean(axis=1)


def detect_swap(pred, target):
    own = np.linalg.norm(pred - target, axis=-1)
    other = np.stack([np.linalg.norm(pred[:, :, n] - target[:, :, 1 - n], axis=-1)
                      for n in range(pred.shape[2])], axis=-1)
    swap_count = (other < own).sum(axis=1)
    swapped = (swap_count >= SWAP_FRAME_TH).any(axis=1)
    return swap_count, swapped


def fold_bounce(pos, vel, limit=LIMIT):
    """单步反弹折叠: pos 沿 vel 走一步, 若越过 [0,limit] 则按反弹规则折回.

    pos (...,2) 左上角坐标, vel (...,2). 返回新位置 (...,2), 新速度 (...,2).
    反弹规则 (与 moving_mnist.py 一致): over -> 2*limit - pos; under -> -pos; 对应轴 vel 取反.
    """
    new_pos = pos + vel
    # 逐轴折叠
    for ax in range(2):
        over = new_pos[..., ax] > limit
        under = new_pos[..., ax] < 0
        new_pos[..., ax] = np.where(over, 2 * limit - new_pos[..., ax], new_pos[..., ax])
        new_pos[..., ax] = np.where(under, -new_pos[..., ax], new_pos[..., ax])
        # 反弹时速度取反 (下一帧用)
        flip = over | under
        vel[..., ax] = np.where(flip, -vel[..., ax], vel[..., ax])
    return new_pos, vel


def extrapolate_ref(centers_obs, k=K_FRAMES):
    """反弹感知外推: 从观测末帧位置沿末帧速度外推 k 步, 逐步反弹折叠.

    centers_obs (B,T,N,2). 返回 ref (B,k,N,2) 未来第 0..k-1 帧的参考位置.
    """
    B, T, N, _ = centers_obs.shape
    q_last = centers_obs[:, T - 1].copy()      # (B,N,2)
    q_prev = centers_obs[:, T - 2]
    vel = q_last - q_prev                       # 末帧速度 (B,N,2)

    ref = np.zeros((B, k, N, 2), dtype=np.float64)
    pos = q_last.copy()
    v = vel.copy()
    for f in range(k):
        pos, v = fold_bounce(pos, v)
        ref[:, f] = pos
    return ref


def planB_deployable(pred, centers_obs, k=K_FRAMES):
    """方案B: 直线外推 ref=q+v (对照)."""
    B, F, N, _ = pred.shape
    T = centers_obs.shape[1]
    out = pred.copy()
    q_last = centers_obs[:, T - 1]
    q_prev = centers_obs[:, T - 2]
    vel = q_last - q_prev
    ref = q_last + vel
    flip_mask = np.zeros(B, dtype=bool)
    for b in range(B):
        kk = min(k, F)
        d00 = np.mean([np.linalg.norm(pred[b, f, 0] - ref[b, 0]) for f in range(kk)])
        d01 = np.mean([np.linalg.norm(pred[b, f, 0] - ref[b, 1]) for f in range(kk)])
        d10 = np.mean([np.linalg.norm(pred[b, f, 1] - ref[b, 0]) for f in range(kk)])
        d11 = np.mean([np.linalg.norm(pred[b, f, 1] - ref[b, 1]) for f in range(kk)])
        if d01 < d00 and d10 < d11:
            flip_mask[b] = True
            out[b] = pred[b][:, ::-1, :]
    return out, flip_mask


def planB2_bounce(pred, centers_obs, k=K_FRAMES):
    """方案B2: 反弹感知外推 ref (逐帧折叠), 前 k 帧平均交叉距离判断交换.

    返回 out (B,F,N,2), flip_mask (B,) bool, ref (B,k,N,2) 供可视化.
    """
    B, F, N, _ = pred.shape
    out = pred.copy()
    ref = extrapolate_ref(centers_obs, k=k)     # (B,k,N,2)
    flip_mask = np.zeros(B, dtype=bool)
    for b in range(B):
        kk = min(k, F)
        d00 = np.mean([np.linalg.norm(pred[b, f, 0] - ref[b, f, 0]) for f in range(kk)])
        d01 = np.mean([np.linalg.norm(pred[b, f, 0] - ref[b, f, 1]) for f in range(kk)])
        d10 = np.mean([np.linalg.norm(pred[b, f, 1] - ref[b, f, 0]) for f in range(kk)])
        d11 = np.mean([np.linalg.norm(pred[b, f, 1] - ref[b, f, 1]) for f in range(kk)])
        if d01 < d00 and d10 < d11:
            flip_mask[b] = True
            out[b] = pred[b][:, ::-1, :]
    return out, flip_mask, ref


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    ds = MovingMNIST(root=DATA_ROOT, train=False, seq_len=N_INPUT + N_FUTURE,
                     length=N_EVAL, seed=TEST_SEED)
    loader = DataLoader(ds, 128, num_workers=2, pin_memory=True, persistent_workers=True)

    pred, target, obs = run_inference(model, loader, device)
    B, F, N, _ = pred.shape
    print(f"\n推理完成: {B} 条序列 x {N} 数字 x {F} 未来帧")

    ade0, fde0 = per_seq_error(pred, target)
    _, swapped0 = detect_swap(pred, target)
    print(f"[基线-不关联]   ADE {ade0.mean():.2f} / FDE {fde0.mean():.2f} px, "
          f"交换序列 {swapped0.sum()}/{B} ({swapped0.sum()/B*100:.1f}%)")

    # 方案B (直线外推, 对照)
    outB, flipB = planB_deployable(pred, obs)
    adeB, fdeB = per_seq_error(outB, target)
    _, swappedB = detect_swap(outB, target)
    flipB_swapped = (flipB & swapped0).sum()
    print(f"[方案B直线外推]  ADE {adeB.mean():.2f} / FDE {fdeB.mean():.2f} px, "
          f"交换 {swappedB.sum()/B*100:.1f}%, 翻转 {flipB.sum()} (准确率 "
          f"{flipB_swapped/flipB.sum()*100:.1f}%)")

    # 方案B2 (反弹折叠外推)
    out2, flip2, ref2 = planB2_bounce(pred, obs)
    ade2, fde2 = per_seq_error(out2, target)
    _, swapped2 = detect_swap(out2, target)
    flip2_swapped = (flip2 & swapped0).sum()
    print(f"[方案B2反弹感知] ADE {ade2.mean():.2f} / FDE {fde2.mean():.2f} px, "
          f"交换 {swapped2.sum()/B*100:.1f}%, 翻转 {flip2.sum()} (准确率 "
          f"{flip2_swapped/flip2.sum()*100:.1f}%)")

    # 报告
    lines = ["# 方案B2: 反弹感知的无真值关联\n\n",
             f"- 模型: {os.path.basename(CKPT_PATH)}  测试集 seed={TEST_SEED} x {B}\n",
             f"- 逻辑: 末帧速度 v=q_last-q_prev 外推, 逐帧撞墙反弹折叠(活动范围 [0,{LIMIT}]), "
             f"前{K_FRAMES}帧平均交叉距离判断交换\n",
             f"- 关键: 反弹折叠是纯几何运算, 无真值, C++ 可移植\n\n",
             "| 方案 | ADE | FDE | 交换序列占比 | 翻转准确率 |\n",
             "|---|---|---|---|---|\n",
             f"| 不关联 | {ade0.mean():.2f} | {fde0.mean():.2f} | {swapped0.sum()/B*100:.1f}% | - |\n",
             f"| 方案B直线外推 | {adeB.mean():.2f} | {fdeB.mean():.2f} | {swappedB.sum()/B*100:.1f}% | "
             f"{flipB_swapped}/{flipB.sum()}={flipB_swapped/flipB.sum()*100:.1f}% |\n",
             f"| 方案B2反弹感知 | {ade2.mean():.2f} | {fde2.mean():.2f} | {swapped2.sum()/B*100:.1f}% | "
             f"{flip2_swapped}/{flip2.sum()}={flip2_swapped/flip2.sum()*100:.1f}% |\n",
             f"\n- ADE 改善 (B2 vs B): {adeB.mean()-ade2.mean():.2f} px\n"]

    with open(os.path.join(OUT_DIR, "关联分析_方案B2.md"), "w", encoding="utf-8") as f:
        f.writelines(lines)

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    names = ["no assoc", "planB (linear)", "planB2 (bounce)"]
    seq_ade = [ade0, adeB, ade2]
    colors = ["gray", "purple", "crimson"]
    sw = swapped0
    for a, n_, c in zip(seq_ade, names, colors):
        axes[0].plot(np.arange(B)[sw], a[sw], ".", ms=4, color=c, alpha=0.5, label=n_)
    axes[0].set_xlabel("swapped sequence index")
    axes[0].set_ylabel("per-sequence ADE (px)")
    axes[0].set_title("ADE on swapped sequences")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    x = np.arange(3)
    axes[1].bar(x, [a.mean() for a in seq_ade], color=colors, alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names)
    axes[1].set_ylabel("mean ADE (px)")
    axes[1].set_title("Overall mean ADE")
    for xi, a in zip(x, seq_ade):
        axes[1].text(xi, a.mean(), f"{a.mean():.2f}", ha="center", va="bottom")
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle("Plan B2: bounce-aware deployable association")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "assoc_planB2.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)

    print(f"\n图已保存 -> {os.path.join(FIG_DIR, 'assoc_planB2.png')}")
    print("报告 ->", os.path.join(OUT_DIR, "关联分析_方案B2.md"))


if __name__ == "__main__":
    main()
