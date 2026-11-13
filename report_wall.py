"""汇报可视化: 大规模样本墙 + 多维分析图.

产出 (results/figs/):
  report_wall.png          — 24样本轨迹预测样本墙(6x4网格)
  report_error_cases.png   — 分类错误案例(4个失败样本, 展示模型边界)
  report_error_dist.png    — ADE误差分布直方图 + 误差vs重叠距离散点图
  report_overlap.png       — 深度重叠场景案例(模型如何应对重叠)

运行: E:\\anaconda3\\envs\\CurveSegmentationMethod\\python.exe report_wall.py
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

CKPT_PATH = r"D:\MovingMNIST\checkpoints\convlstm_v5_anchor_30ep_best.pth"
DATA_ROOT = r"D:\MovingMNIST\dataset\data"
FIG_DIR = r"D:\MovingMNIST\results\figs"
TEST_SEED = 2026


def load_model(device):
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    model = MultiTaskConvLSTMv2(cls_mode="gap", traj_mode="delta", detach_anchor=False,
                                h1_ch=128, h2_ch=256, dec_hidden=768,
                                slot_feat=True, slot_anchor=True).to(device)
    sd = model.state_dict()
    for k, v in ckpt["model_state_dict"].items():
        if k in sd and sd[k].shape == v.shape:
            sd[k] = v
        elif k in sd and "cell.weight_ih" in k and sd[k].shape[1] > v.shape[1]:
            sd[k][:, :v.shape[1]] = v
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def predict_batch(model, device, ds, indices):
    """对指定索引的样本批量预测, 返回结果字典列表."""
    results = []
    for idx in indices:
        s = ds[idx]
        video = s["video"].unsqueeze(0).to(device)
        labels = s["labels"].unsqueeze(0).to(device)
        centers = s["centers"].unsqueeze(0).to(device)
        labels, centers = sort_slots(labels, centers)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            obs_centers = centers[:, :N_INPUT] / CANVAS
            logits, traj, _, _ = model(video[:, :N_INPUT], obs_centers)
        pred = (traj[0, 1:].detach().float() * CANVAS).cpu().numpy()  # (F,N,2)
        targ = centers[0, N_INPUT:].cpu().numpy()
        pc = logits[0].detach().argmax(-1).cpu().tolist()
        tc = labels[0].cpu().tolist()
        own = np.linalg.norm(pred - targ, axis=-1)
        ade = own.mean()
        results.append({
            "idx": idx,
            "video": video[0].cpu().numpy(),
            "centers": centers[0].cpu().numpy(),
            "pred": pred, "targ": targ,
            "pred_cls": pc, "true_cls": tc, "ade": ade,
        })
    return results


def draw_traj_axis(ax, r, show_title=True, title_color="black"):
    """在单个子图上画轨迹预测(观测/真值/预测)."""
    video = r["video"]; centers = r["centers"]; pred = r["pred"]
    ax.imshow(video[N_INPUT - 1, 0], cmap="gray", vmin=0, vmax=1)
    for n, mk in enumerate(["o", "s"]):
        ax.plot(centers[:N_INPUT, n, 0], centers[:N_INPUT, n, 1], "-" + mk,
                color="white", ms=2.5, lw=1.5)
        ax.plot(centers[N_INPUT:, n, 0], centers[N_INPUT:, n, 1], "-" + mk,
                color="lime", ms=2.5, lw=1.5)
        ax.plot(pred[:, n, 0], pred[:, n, 1], "--x", color="red", ms=3, lw=1.5)
    ax.axis("off")
    if show_title:
        ok = "对" if r["pred_cls"] == r["true_cls"] else "错"
        ax.set_title(f"#{r['idx']} 预{r['pred_cls']}/真{r['true_cls']}({ok}) ADE{r['ade']:.1f}",
                     fontsize=8, color=title_color)


def report_wall(results):
    """图1: 24样本墙 (6x4)."""
    n = len(results)
    cols, rows = 6, 4
    fig, axes = plt.subplots(rows, cols, figsize=(20, 13))
    for i, (r, ax) in enumerate(zip(results, axes.flat)):
        draw_traj_axis(ax, r)
    fig.suptitle("轨迹预测样本墙（24样本：白=观测 / 绿=真值 / 红=预测）", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(FIG_DIR, "report_wall.png")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("样本墙已保存 ->", p)


def report_error_cases(results):
    """图2: 分类错误案例 (展示模型边界)."""
    errs = [r for r in results if r["pred_cls"] != r["true_cls"]]
    if len(errs) < 4:
        print("分类错误样本不足4个, 跳过错误案例图")
        return
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    for ax, r in zip(axes, errs[:4]):
        draw_traj_axis(ax, r, title_color="#c0392b")
    fig.suptitle("分类错误案例（模型在哪些情况下出错？红色=预测 / 绿色=真值）",
                 fontsize=15, y=1.0)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "report_error_cases.png")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("错误案例图已保存 ->", p)


def report_error_dist(model, device, ds, n=500):
    """图3: ADE误差分布 + 误差vs重叠距离散点."""
    ades, dists = [], []
    with torch.no_grad():
        loader = DataLoader(ds, 128, num_workers=2, pin_memory=True)
        for b in loader:
            v = b["video"].cuda(); c = b["centers"].cuda(); lab = b["labels"].cuda()
            lab, c = sort_slots(lab, c)
            oc = c[:, :N_INPUT] / CANVAS
            _, traj, _, _ = model(v[:, :N_INPUT], oc)
            pred = (traj[:, 1:].float() * CANVAS).cpu().numpy()
            targ = c[:, N_INPUT:].cpu().numpy()
            own = np.linalg.norm(pred - targ, axis=-1)  # (B,F,N)
            ades.extend(own.mean(axis=(1, 2)))
            # 观测末帧重叠距离
            d = np.linalg.norm(c[:, N_INPUT-1, 0].cpu().numpy() - c[:, N_INPUT-1, 1].cpu().numpy(), axis=-1)
            dists.extend(d)

    ades = np.array(ades); dists = np.array(dists)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(ades, bins=40, color="#4c7ac0", alpha=0.85, edgecolor="white")
    ax.axvline(ades.mean(), color="#d43a3a", ls="--", lw=2, label=f"均值 {ades.mean():.2f}px")
    ax.axvline(np.median(ades), color="#2e8b57", ls=":", lw=2, label=f"中位数 {np.median(ades):.2f}px")
    ax.set_xlabel("轨迹预测误差 ADE (px)")
    ax.set_ylabel("样本数")
    ax.set_title(f"ADE 误差分布（{len(ades)} 样本）")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.scatter(dists, ades, s=12, alpha=0.4, color="#4c7ac0")
    ax.axvline(14, color="#d43a3a", ls="--", lw=1.5, label="深度重叠阈值 14px")
    ax.set_xlabel("观测末帧两数字间距 (px)")
    ax.set_ylabel("ADE (px)")
    ax.set_title("误差 vs 观测末帧重叠程度")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle("模型误差分布：绝大多数样本 ADE 集中在 2~6px，重叠越深误差越大",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "report_error_dist.png")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("误差分布图已保存 ->", p)


def report_overlap(results):
    """图4: 深度重叠场景案例."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    for ax, r in zip(axes, results[:4]):
        draw_traj_axis(ax, r)
    fig.suptitle("深度重叠场景：两数字重叠后分开，模型能否正确跟踪各自身份？",
                 fontsize=15, y=1.0)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "report_overlap.png")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("重叠案例图已保存 ->", p)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    ds = MovingMNIST(root=DATA_ROOT, train=False, seq_len=N_INPUT + N_FUTURE,
                     length=500, seed=TEST_SEED)

    # 1. 样本墙: 24个随机样本
    wall_idx = list(range(0, 48, 2))  # 均匀抽24个
    results_wall = predict_batch(model, device, ds, wall_idx)
    report_wall(results_wall)

    # 2. 分类错误案例: 已知错误样本
    err_idx = [97, 111, 117, 118, 124, 130, 150, 159, 178, 227, 239, 265]
    results_err = predict_batch(model, device, ds, err_idx)
    report_error_cases(results_err)

    # 3. 误差分布 (全500样本)
    report_error_dist(model, device, ds, n=500)

    # 4. 深度重叠案例
    overlap_idx = [12, 14, 15, 16]
    results_overlap = predict_batch(model, device, ds, overlap_idx)
    report_overlap(results_overlap)

    print("\n所有多维可视化图已生成。")


if __name__ == "__main__":
    main()
