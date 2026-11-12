"""方案 B (可部署版): 无真值的显式关联 —— 用观测段运动信息纠正深度重叠时的槽位交换.

核心突破 (前面实验的结论):
  - 纯位置关联失败: 重叠时位置无信息.
  - 嵌入关联失败: 嵌入坍塌 + 共享编码器下身份区分污染主任务.
  - 方案B正解: 网络保持纯 v4ft (不改), 关联完全在网络外做, 用"观测段(前10帧)真实位置
    + 速度外推"作为参考 (这些推理时可得, 不需要未来真值), 判断并纠正交换.

无真值判断逻辑 (纯推理可得, C++ 可移植):
  1. 观测段每数字位置已知 (输入视频质心), 末帧位置 q0,q1, 末帧速度 v0,v1.
  2. 速度外推参考: ref0 = q0 + v0, ref1 = q1 + v1 (下一帧该在哪).
  3. 模型预测未来轨迹 pred (F,N,2), 第0帧 p0,p1.
  4. 若 p0 更接近 ref1 且 p1 更接近 ref0 (交叉匹配), 则模型把轨迹接反 -> 翻转整段.
  5. 为降漏判: 用前 K 帧的平均交叉距离判断, 而非仅第0帧.

产出: 终端报告 + results/figs/assoc_planB.png + results/关联分析_方案B.md

运行: E:\\anaconda3\\envs\\CurveSegmentationMethod\\python.exe assoc_planB.py
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
K_FRAMES = 5          # 用前 K 帧平均判断交换 (降漏判)
SWAP_FRAME_TH = 3


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
    """返回 pred (B,F,N,2) 未来轨迹 px, target (B,F,N,2), centers_obs (B,T,N,2) 观测段质心."""
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


def planB_deployable(pred, centers_obs, k=K_FRAMES):
    """无真值可部署关联: 用观测段末帧位置+速度外推判断交换并纠正.

    pred (B,F,N,2) 未来轨迹预测, centers_obs (B,T,N,2) 观测段质心(推理可得).
    返回 out (B,F,N,2), flip_mask (B,) bool."""
    B, F, N, _ = pred.shape
    T = centers_obs.shape[1]
    out = pred.copy()

    # 观测末帧位置 + 末帧速度 (从观测段最后两帧)
    q_last = centers_obs[:, T - 1]          # (B,N,2)
    q_prev = centers_obs[:, T - 2]          # (B,N,2)
    vel = q_last - q_prev                    # (B,N,2) 末帧速度
    ref = q_last + vel                       # (B,N,2) 速度外推参考(下一帧该在哪)

    flip_mask = np.zeros(B, dtype=bool)

    for b in range(B):
        # 用前 k 帧的平均交叉距离判断
        kk = min(k, F)
        # 槽位0预测 离 ref0 近还是 ref1 近 (平均)
        d00 = np.mean([np.linalg.norm(pred[b, f, 0] - ref[b, 0]) for f in range(kk)])
        d01 = np.mean([np.linalg.norm(pred[b, f, 0] - ref[b, 1]) for f in range(kk)])
        d10 = np.mean([np.linalg.norm(pred[b, f, 1] - ref[b, 0]) for f in range(kk)])
        d11 = np.mean([np.linalg.norm(pred[b, f, 1] - ref[b, 1]) for f in range(kk)])
        # 交叉匹配 -> 交换
        if d01 < d00 and d10 < d11:
            flip_mask[b] = True
            out[b] = pred[b][:, ::-1, :]     # 翻转槽位0/1

    return out, flip_mask


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
    print(f"[基线-不关联]       ADE {ade0.mean():.2f} / FDE {fde0.mean():.2f} px, "
          f"交换序列 {swapped0.sum()}/{B} ({swapped0.sum()/B*100:.1f}%)")

    out, flip_mask = planB_deployable(pred, obs)
    ade1, fde1 = per_seq_error(out, target)
    _, swapped1 = detect_swap(out, target)
    # 翻转准确率: 翻转的序列里, 真正交换的比例
    flip_swapped = (flip_mask & swapped0).sum()
    flip_total = flip_mask.sum()
    print(f"[方案B无真值关联]   ADE {ade1.mean():.2f} / FDE {fde1.mean():.2f} px, "
          f"交换序列 {swapped1.sum()}/{B} ({swapped1.sum()/B*100:.1f}%)")
    print(f"  翻转 {flip_total} 序列, 其中真交换 {flip_swapped} (准确率 "
          f"{flip_swapped/flip_total*100:.1f}%), 误判 {flip_total-flip_swapped}")

    # 报告
    lines = ["# 方案B: 无真值显式关联 (可部署版)\n\n",
             f"- 模型: {os.path.basename(CKPT_PATH)}  测试集 seed={TEST_SEED} x {B}\n",
             f"- 逻辑: 观测段末帧位置+速度外推(ref=q+v) 作为参考, 前{K_FRAMES}帧平均交叉距离判断交换\n",
             f"- 关键: 无真值, 纯推理可得, C++ 可移植\n\n",
             "| 方案 | ADE | FDE | 交换序列占比 |\n",
             "|---|---|---|---|\n",
             f"| 不关联 | {ade0.mean():.2f} | {fde0.mean():.2f} | {swapped0.sum()/B*100:.1f}% |\n",
             f"| 方案B无真值 | {ade1.mean():.2f} | {fde1.mean():.2f} | {swapped1.sum()/B*100:.1f}% |\n",
             f"\n- ADE 改善: {ade0.mean()-ade1.mean():.2f} px "
             f"(翻转准确率 {flip_swapped}/{flip_total} = {flip_swapped/flip_total*100:.1f}%)\n"]

    with open(os.path.join(OUT_DIR, "关联分析_方案B.md"), "w", encoding="utf-8") as f:
        f.writelines(lines)

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    seq_ade = [ade0, ade1]
    names = ["no assoc", "planB (deployable)"]
    colors = ["gray", "purple"]
    sw = swapped0
    for a, n_, c in zip(seq_ade, names, colors):
        axes[0].plot(np.arange(B)[sw], a[sw], ".", ms=4, color=c, alpha=0.5, label=n_)
    axes[0].set_xlabel("swapped sequence index")
    axes[0].set_ylabel("per-sequence ADE (px)")
    axes[0].set_title("ADE on swapped sequences")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    x = np.arange(2)
    axes[1].bar(x, [a.mean() for a in seq_ade], color=colors, alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names)
    axes[1].set_ylabel("mean ADE (px)")
    axes[1].set_title("Overall mean ADE")
    for xi, a in zip(x, seq_ade):
        axes[1].text(xi, a.mean(), f"{a.mean():.2f}", ha="center", va="bottom")
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle("Plan B: non-learned deployable association (velocity extrapolation)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "assoc_planB.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)

    print(f"\n图已保存 -> {os.path.join(FIG_DIR, 'assoc_planB.png')}")
    print("报告 ->", os.path.join(OUT_DIR, "关联分析_方案B.md"))


if __name__ == "__main__":
    main()
