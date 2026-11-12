"""汇报可视化: 生成高质量成果展示图.

产出 (results/figs/report_*):
  1. report_traj.png      — v5_anchor 轨迹预测对比 (4样本: 观测/真值/模型/线性基线)
  2. report_progress.png  — 各版本指标进步曲线 (ADE/FDE/分类 随版本演进)
  3. report_swap.png      — 交换根因可视化 (交换率 vs 重叠距离)

运行: E:\\anaconda3\\envs\\CurveSegmentationMethod\\python.exe report_visuals.py
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
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

CKPT_PATH = r"D:\MovingMNIST\checkpoints\convlstm_v5_anchor_best.pth"
DATA_ROOT = r"D:\MovingMNIST\dataset\data"
FIG_DIR = r"D:\MovingMNIST\results\figs"
TEST_SEED = 2026


def load_model(device):
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    cfg = ckpt["config"]
    model = MultiTaskConvLSTMv2(cls_mode="gap", traj_mode="delta", detach_anchor=False,
                                h1_ch=128, h2_ch=256, dec_hidden=768,
                                slot_feat=True, slot_anchor=True).to(device)
    # 迁移加载
    sd = model.state_dict()
    for k, v in ckpt["model_state_dict"].items():
        if k in sd and sd[k].shape == v.shape:
            sd[k] = v
        elif k in sd and "cell.weight_ih" in k and sd[k].shape[1] > v.shape[1]:
            sd[k][:, :v.shape[1]] = v
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def report_traj(model, device):
    """图1: v5_anchor 轨迹预测对比 (挑4个有代表性的样本)."""
    ds = MovingMNIST(root=DATA_ROOT, train=False, seq_len=N_INPUT + N_FUTURE,
                     length=50, seed=TEST_SEED)
    # 挑样本: 有重叠、有反弹、有交换的多样本
    samples = [3, 7, 15, 22]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    for i, idx in enumerate(samples):
        s = ds[idx]
        video = s["video"].unsqueeze(0).to(device)
        labels = s["labels"].unsqueeze(0).to(device)
        centers = s["centers"].unsqueeze(0).to(device)
        labels, centers = sort_slots(labels, centers)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            obs_centers = centers[:, :N_INPUT] / CANVAS
            logits, traj, _ = model(video[:, :N_INPUT], obs_centers)
        pred = (traj[0, 1:].detach().float() * CANVAS).cpu().numpy()   # (F,N,2)
        centers = centers[0].cpu().numpy()
        video_np = video[0].cpu().numpy()

        ax = axes[i]
        # 背景用最后一帧观测 (第9帧)
        ax.imshow(video_np[N_INPUT - 1, 0], cmap="gray", vmin=0, vmax=1)
        for n, (mk, name) in enumerate(zip(["o", "s"], ["数字1", "数字2"])):
            obs = centers[:N_INPUT, n]
            fut = centers[N_INPUT:, n]
            p = pred[:, n]
            ax.plot(obs[:, 0], obs[:, 1], "-" + mk, color="white", ms=5, lw=2,
                    label="观测" if n == 0 else None)
            ax.plot(fut[:, 0], fut[:, 1], "-" + mk, color="lime", ms=5, lw=2,
                    label="真值" if n == 0 else None)
            ax.plot(p[:, 0], p[:, 1], "--x", color="red", ms=6, lw=2,
                    label="预测" if n == 0 else None)
        pred_cls = logits[0].detach().argmax(-1).cpu().tolist()
        true_cls = labels[0].cpu().tolist()
        ax.set_title(f"样本{idx}\n真值类别{true_cls} / 预测{['%d' % c for c in pred_cls]}",
                     fontsize=12)
        ax.axis("off")
    axes[0].legend(loc="upper right", fontsize=11, framealpha=0.9)
    fig.suptitle("v5_anchor 轨迹预测结果（白=观测10帧 / 绿=真实未来10帧 / 红=模型预测10帧）",
                 fontsize=15, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(FIG_DIR, "report_traj.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("图1 已保存 ->", p)


def report_progress():
    """图2: 各版本指标进步曲线."""
    versions = ["v1", "v2a", "v3", "v4", "v4ft", "v5_anchor"]
    ade = [7.81, 6.05, 6.02, 5.27, 5.25, 4.99]
    fde = [9.98, 8.35, 8.30, 7.40, 7.28, 7.00]
    acc = [95.6, 95.5, 96.8, 96.2, 96.5, 96.35]
    baseline_ade = 23.88

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    x = np.arange(len(versions))
    colors = ["#b0b0b0", "#b0b0b0", "#7aa0d0", "#7aa0d0", "#4c7ac0", "#d43a3a"]

    ax = axes[0]
    ax.axhline(baseline_ade, color="gray", ls="--", lw=1, label=f"线性基线 ADE={baseline_ade}")
    ax.plot(x, ade, "o-", color="#d43a3a", lw=2, ms=8, label="ADE (px)")
    ax.plot(x, fde, "s-", color="#4c7ac0", lw=2, ms=8, label="FDE (px)")
    ax.set_xticks(x)
    ax.set_xticklabels(versions)
    ax.set_ylabel("误差 (px)")
    ax.set_title("轨迹预测误差演进（越低越好）")
    ax.grid(alpha=0.3)
    ax.legend()
    for xi, v in zip(x, ade):
        ax.text(xi, v - 1.5, f"{v:.2f}", ha="center", fontsize=9, color="#d43a3a")

    ax = axes[1]
    ax.plot(x, acc, "o-", color="#2e8b57", lw=2, ms=8, label="分类准确率")
    ax.axhline(99.0, color="red", ls="--", lw=1, label="目标 99%")
    ax.axhline(96.5, color="gray", ls=":", lw=1, label="当前 96.35%")
    ax.set_xticks(x)
    ax.set_xticklabels(versions)
    ax.set_ylabel("分类准确率 (%)")
    ax.set_ylim(70, 100)
    ax.set_title("分类准确率演进")
    ax.grid(alpha=0.3)
    ax.legend()
    for xi, v in zip(x, acc):
        ax.text(xi, v + 1.2, f"{v:.1f}", ha="center", fontsize=9)

    fig.suptitle("模型迭代成果：ADE 首次破 5.0，分类稳定 96%+", fontsize=15, y=1.02)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "report_progress.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("图2 已保存 ->", p)


def report_swap():
    """图3: 交换根因可视化 (交换率 vs 观测末帧重叠距离)."""
    # 数据来自诊断 (diag_swap_origin.py 结论)
    bins = ["<14px\n(深度重叠)", "14~28px", "≥28px"]
    swap_rate = [45.5, 18.3, 15.9]
    counts = [651, 1022, 327]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    bars = ax.bar(bins, swap_rate, color=["#d43a3a", "#e8a03a", "#2e8b57"], alpha=0.85)
    ax.set_ylabel("交换率 (%)")
    ax.set_title("槽位交换率 vs 观测末帧两数字距离")
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, swap_rate):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%", ha="center", fontsize=12)

    # 交换起点分布
    ax = axes[1]
    frames = np.arange(10)
    # 数据来自 diag_swap_origin.py: 第0帧47.3%, 后续递减
    start_dist = [47.3, 7.1, 5.4, 6.4, 6.9, 9.2, 10.3, 7.5, 0, 0]
    ax.bar(frames, start_dist, color="#4c7ac0", alpha=0.85)
    ax.set_xlabel("交换起始帧 (未来第几帧)")
    ax.set_ylabel("占比 (%)")
    ax.set_title("交换起始帧分布（47.3% 在首帧就交换）")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("交换根因：重叠时身份丢失，分开时随机接错", fontsize=15, y=1.02)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "report_swap.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("图3 已保存 ->", p)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model = load_model(device)
    report_traj(model, device)
    report_progress()
    report_swap()
    print("\n所有汇报图已生成。")


if __name__ == "__main__":
    main()
