"""第三步: 轨迹级位置关联 (卡尔曼滤波 + 速度连续性) —— 无嵌入, 纯位置解决槽位交换.

背景:
  方向2 结论: 不用身份嵌入 (嵌入已证明坍塌, 且"类别≠身份"语义错), 改用纯位置方法.
  误差分析已定位: 深度重叠时槽位身份交换是主要误差来源 (25.1% 序列).

核心思路 (对齐工业界 MOT 的卡尔曼 + 数据关联, 但针对 2 数字场景简化):
  1. 对每个槽位, 用一个"恒速度卡尔曼滤波器"维持 [位置x, 位置y, 速度vx, 速度vy] 状态.
  2. 逐未来帧:
     - 若两槽位预测距离 >= 阈值 (不重叠): 用卡尔曼更新各自状态, 身份明确.
     - 若 < 阈值 (重叠/接近, 身份可能混淆): 暂停身份更新, 进入"模糊区",
       但继续用卡尔曼"预测"阶段外推两个轨迹 (不做观测更新, 保持各自历史速度).
  3. 当模糊区结束 (两槽位重新分开):
     - 用"速度连续性 + 位置连续性"判断当前两预测轨迹该接哪个历史槽位:
       把当前帧预测位置, 与两个"卡尔曼外推位置"做匈牙利匹配 (代价=位置距离).
     - 若匹配结果指向交换, 则从该帧起整体翻转槽位身份, 并锁定.
  4. 关键: 关联是"轨迹级一次性翻转", 不是逐帧重排, 保持时间连续性.

与 v2 的区别:
  - v2 用嵌入判断交换, 但嵌入坍塌 → 失败.
  - 本脚本纯位置: 用卡尔曼"速度外推"作为参考身份, 语义正确 (参考身份=轨迹的运动状态).

产出:
  终端报告 + results/figs/assoc_kalman.png
  报告写入 results/关联分析.md (追加卡尔曼方案)

运行:
  E:\\anaconda3\\envs\\CurveSegmentationMethod\\python.exe assoc_kalman.py
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moving_mnist import MovingMNIST
from train_convlstm_v2 import MultiTaskConvLSTMv2, sort_slots, N_INPUT, N_FUTURE, CANVAS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_ROOT = r"D:\MovingMNIST\dataset\data"
CKPT_PATH = r"D:\MovingMNIST\checkpoints\convlstm_v4ft_embed_best.pth"
OUT_DIR = r"D:\MovingMNIST\results"
FIG_DIR = os.path.join(OUT_DIR, "figs")
TEST_SEED = 2026
N_EVAL = 2000
SWAP_FRAME_TH = 3
OVERLAP_TH = 14.0      # 重叠判定阈值 (px)
Q_POS = 1e-2           # 卡尔曼过程噪声 (位置)
Q_VEL = 1e-1           # 卡尔曼过程噪声 (速度)
R_POS = 1.0            # 观测噪声 (位置)


class ConstantVelocityKF:
    """恒速度卡尔曼滤波器, 状态 [x, y, vx, vy], 观测 [x, y]. 2 数字各自独立."""

    def __init__(self, x0, y0, vx0=0.0, vy0=0.0):
        self.x = np.array([x0, y0, vx0, vy0], dtype=np.float64)
        # 初始协方差
        self.P = np.eye(4) * 1.0
        # 状态转移 F: x += vx*dt, y += vy*dt (dt=1)
        self.F = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float64)
        # 观测矩阵 H: 观测 [x, y]
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float64)
        # 过程噪声 Q, 观测噪声 R
        self.Q = np.diag([Q_POS, Q_POS, Q_VEL, Q_VEL])
        self.R = np.eye(2) * R_POS

    def predict(self):
        """预测阶段: 外推一步 (只更新状态与协方差, 不做观测)."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].copy()

    def update(self, z):
        """更新阶段: 用观测 z=[x,y] 修正状态."""
        z = np.asarray(z, dtype=np.float64)
        y = z - self.H @ self.x           # 残差
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def pos(self):
        return self.x[:2].copy()

    def vel(self):
        return self.x[2:].copy()


def load_embed_model(device):
    """加载模型 (embed=True 因为 checkpoint 是带嵌入头的)."""
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    cfg = ckpt["config"]
    model = MultiTaskConvLSTMv2(cls_mode=cfg["cls_mode"], traj_mode=cfg["traj_mode"],
                                detach_anchor=cfg["detach_anchor"], h1_ch=cfg["h1_ch"],
                                h2_ch=cfg["h2_ch"], dec_hidden=cfg["dec_hidden"],
                                slot_feat=cfg.get("slot_feat", False),
                                embed=cfg.get("embed", False)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def run_inference(model, loader, device):
    """返回 pred (B,F,N,2) px, target (B,F,N,2)."""
    preds, targets = [], []
    for batch in loader:
        video = batch["video"].to(device)
        centers = batch["centers"].to(device)
        _, centers = sort_slots(batch["labels"].to(device), centers)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits, traj, _, _ = model(video[:, :N_INPUT])
        preds.append((traj[:, 1:].float() * CANVAS).cpu().numpy())
        targets.append(centers[:, N_INPUT:].cpu().numpy())
    return np.concatenate(preds, 0), np.concatenate(targets, 0)


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


def kalman_associate(pred, overlap_th=OVERLAP_TH):
    """轨迹级位置关联 (卡尔曼 + 速度连续性).

    pred: (B,F,N,2) 预测位置 (px), N=2.
    对每条序列: 用两个恒速度 KF 维持槽位 0/1 的轨迹;
    重叠时暂停更新(只 predict 外推), 分开时用外推位置做匈牙利匹配, 判断是否交换.
    返回 out (B,F,N,2) 关联后预测, 以及统计."""
    B, F, N, _ = pred.shape
    assert N == 2, "本脚本针对 2 数字场景"
    out = pred.copy()
    n_swap_seq = 0
    n_swap_frame = 0

    for b in range(B):
        # 初始化: 用首帧预测位置 + 首两帧速度初始化两个 KF
        kf = []
        for n in range(N):
            p0 = pred[b, 0, n]
            if F >= 2:
                v = pred[b, 1, n] - pred[b, 0, n]
            else:
                v = np.zeros(2)
            kf.append(ConstantVelocityKF(p0[0], p0[1], v[0], v[1]))

        swapped = False
        swap_frame = -1

        for f in range(F):
            # 当前帧两槽位预测距离
            dist = np.linalg.norm(pred[b, f, 0] - pred[b, f, 1])

            if not swapped:
                if dist >= overlap_th:
                    # 不重叠: 正常用观测更新各自 KF
                    for n in range(N):
                        kf[n].update(pred[b, f, n])
                else:
                    # 重叠: 暂停更新, 只做 predict 外推 (保持历史速度, 不吸收混淆观测)
                    # 记录进入模糊区时的外推位置, 供后续匹配参考
                    for n in range(N):
                        kf[n].predict()
            else:
                # 已交换: 只需持续翻转, 不再更新 KF (身份已锁)
                pass

            # 判断是否需要交换: 当从重叠状态"即将/已经分开"时, 用外推位置判断
            # 简化: 在重叠帧内, 若"当前槽位0 更接近 槽位1 的外推位置" 且反之, 判定交换
            if not swapped and dist < overlap_th:
                p0_kf, p1_kf = kf[0].pos(), kf[1].pos()
                cur0, cur1 = pred[b, f, 0], pred[b, f, 1]
                # 匈牙利匹配: 当前两预测 vs 两外推参考位置
                cost = np.array([[np.linalg.norm(cur0 - p0_kf), np.linalg.norm(cur0 - p1_kf)],
                                 [np.linalg.norm(cur1 - p0_kf), np.linalg.norm(cur1 - p1_kf)]])
                row, col = linear_sum_assignment(cost)
                # col[i] 表示当前槽位 i 应接到参考槽位 col[i]; 若 col=[1,0] 则交换
                if col[0] == 1 and col[1] == 0:
                    swapped = True
                    swap_frame = f

        if swapped:
            n_swap_seq += 1
            for f in range(swap_frame, F):
                out[b, f] = pred[b, f][::-1]
                n_swap_frame += 1

    return out, n_swap_seq, n_swap_frame


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_embed_model(device)
    ds = MovingMNIST(root=DATA_ROOT, train=False, seq_len=N_INPUT + N_FUTURE,
                     length=N_EVAL, seed=TEST_SEED)
    loader = DataLoader(ds, 128, num_workers=2, pin_memory=True, persistent_workers=True)

    pred, target = run_inference(model, loader, device)
    B, F, N, _ = pred.shape
    print(f"\n推理完成: {B} 条序列 x {N} 数字 x {F} 未来帧")

    ade0, fde0 = per_seq_error(pred, target)
    _, swapped0 = detect_swap(pred, target)
    print(f"[基线-不关联]       ADE {ade0.mean():.2f} / FDE {fde0.mean():.2f} px, "
          f"交换序列 {swapped0.sum()}/{B} ({swapped0.sum()/B*100:.1f}%)")

    out, nseq, nframe = kalman_associate(pred)
    ade1, fde1 = per_seq_error(out, target)
    _, swapped1 = detect_swap(out, target)
    print(f"[卡尔曼位置关联]     ADE {ade1.mean():.2f} / FDE {fde1.mean():.2f} px, "
          f"交换序列 {swapped1.sum()}/{B} ({swapped1.sum()/B*100:.1f}%), "
          f"触发交换 {nseq} 序列 / {nframe} 帧")

    # 报告
    lines = ["\n## 卡尔曼位置关联 (方向2, 无嵌入)\n\n",
             f"- 重叠阈值 {OVERLAP_TH}px, 恒速度卡尔曼 (Q_pos={Q_POS}, Q_vel={Q_VEL}, R={R_POS})\n",
             f"- 关联策略: 重叠时暂停观测更新只外推, 分开时用外推位置匈牙利匹配, 一次性翻转锁定\n\n",
             "| 方案 | ADE (px) | FDE (px) | 交换序列占比 |\n",
             "|---|---|---|---|\n",
             f"| 不关联 | {ade0.mean():.2f} | {fde0.mean():.2f} | {swapped0.sum()/B*100:.1f}% |\n",
             f"| 卡尔曼关联 | {ade1.mean():.2f} | {fde1.mean():.2f} | {swapped1.sum()/B*100:.1f}% |\n",
             f"\n- ADE 改善: {ade0.mean()-ade1.mean():.2f} px, 交换序列变化: "
             f"{swapped0.sum()/B*100:.1f}% -> {swapped1.sum()/B*100:.1f}%\n"]

    with open(os.path.join(OUT_DIR, "关联分析.md"), "a", encoding="utf-8") as f:
        f.writelines(lines)

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    seq_ade = [ade0, ade1]
    names = ["no assoc", "kalman assoc"]
    colors = ["gray", "blue"]
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
    fig.suptitle("Kalman track-level association (position-only, no embedding)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "assoc_kalman.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)

    print(f"\n图已保存 -> {os.path.join(FIG_DIR, 'assoc_kalman.png')}")
    print("报告追加 ->", os.path.join(OUT_DIR, "关联分析.md"))


if __name__ == "__main__":
    main()
