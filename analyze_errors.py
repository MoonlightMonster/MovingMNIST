"""误差分解分析: v3 模型的轨迹/分类误差到底来自哪里.

回答三个问题 (修复方法完全不同, 先分解再决定结构改动):
  1. 撞墙反弹贡献多少误差?   —— 从真值轨迹检测反弹帧(速度变号), 对比反弹帧 vs 平稳帧误差
  2. 误差是否被少数灾难样本(槽位交换)拉高? —— 每序列 ADE 分位数 + top 贡献占比 + 交换检测
  3. 分类错误是否集中在数字重叠样本? —— 重叠/非重叠序列的准确率对比

产出:
  终端文字报告 + results/figs/analysis_error_dist.png (误差分布)
                 + results/figs/analysis_bounce.png    (反弹帧 vs 平稳帧逐帧误差)
                 + results/figs/analysis_worst.png     (误差最大的 6 条样本)
  结论汇总追加到 results/误差分析.md

运行:
  E:\\anaconda3\\envs\\CurveSegmentationMethod\\python.exe analyze_errors.py
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 须在 import torch 前设置

import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moving_mnist import MovingMNIST
from train_convlstm_v2 import sort_slots, N_INPUT, N_FUTURE, CANVAS
from visualize_results import load_model, draw_tracks, CKPT_PATH

import re
_tag_m = re.search(r"convlstm_(\w+?)_best", os.path.basename(CKPT_PATH))
TAG = _tag_m.group(1) if _tag_m else "model"   # 从 checkpoint 名推出版本号, 报告按版本命名

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_ROOT = r"D:\MovingMNIST\dataset\data"
OUT_DIR = r"D:\MovingMNIST\results"
FIG_DIR = os.path.join(OUT_DIR, "figs")
TEST_SEED = 2026
N_EVAL = 2000          # 与正式测试同口径
DIGIT_SIZE = 28
SWAP_FRAME_TH = 3      # 一条序列中预测离"另一个数字"更近的帧数 >= 此值则判为槽位交换


@torch.no_grad()
def run_inference(model, loader, device):
    """返回每序列的 err (B,F,N), pred/target/logits/labels/centers (已排序槽位)."""
    errs, preds, targets, all_logits, all_labels, all_centers = [], [], [], [], [], []
    for batch in loader:
        video = batch["video"].to(device)
        labels = batch["labels"].to(device)
        centers = batch["centers"].to(device)
        labels, centers = sort_slots(labels, centers)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits, traj, _ = model(video[:, :N_INPUT])
        pred = traj[:, 1:].float() * CANVAS            # (B,F,N,2) px
        target = centers[:, N_INPUT:]
        errs.append(((pred - target).norm(dim=-1)).cpu())       # (B,F,N)
        preds.append(pred.cpu())
        targets.append(target.cpu())
        all_logits.append(logits.float().cpu())
        all_labels.append(labels.cpu())
        all_centers.append(centers.cpu())
    return (torch.cat(errs).numpy(), torch.cat(preds).numpy(), torch.cat(targets).numpy(),
            torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy(),
            torch.cat(all_centers).numpy())


def bounce_mask_future(centers):
    """检测未来每帧是否发生反弹: 速度符号翻转 (任一轴).
    centers: (B,T,N,2) -> (B,F,N) bool, 第 f 项对应时刻 t=N_INPUT+f.
    最后一帧的反弹需要 t+1 的速度, 无法检测, 恒为 False."""
    vel = np.diff(centers, axis=1)                     # (B,T-1,N,2), vel[:,j] = c[j+1]-c[j]
    flip = (vel[:, 1:] * vel[:, :-1] < 0).any(axis=-1)  # (B,T-2,N), 下标 k => 帧 k+1 是反弹点
    B, _, N = flip.shape
    mask = np.zeros((B, N_FUTURE, N), dtype=bool)
    # 未来帧 t = N_INPUT+f (f=0..F-2) 对应 flip 下标 k = t-1
    mask[:, :N_FUTURE - 1] = flip[:, N_INPUT - 1:N_INPUT - 1 + N_FUTURE - 1]
    return mask


def min_center_dist_input(centers):
    """输入窗口 (前10帧) 内两数字质心的最小欧氏距离 (px).
    centers: (B,T,N,2) -> (B,) float. 数字 28x28, 距离<28 包围盒才相交, <14 为深度重叠."""
    c = centers[:, :N_INPUT]                           # (B,I,N,2)
    d = c[:, :, 0] - c[:, :, 1]                        # (B,I,2)
    return np.linalg.norm(d, axis=-1).min(axis=1)      # (B,)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    ds = MovingMNIST(root=DATA_ROOT, train=False, seq_len=N_INPUT + N_FUTURE,
                     length=N_EVAL, seed=TEST_SEED)
    loader = DataLoader(ds, 128, num_workers=2, pin_memory=True, persistent_workers=True)

    err, pred, target, logits, labels, centers = run_inference(model, loader, device)
    B, F, N = err.shape
    print(f"\n推理完成: {B} 条序列 x {N} 数字 x {F} 未来帧")
    print(f"总体 ADE {err.mean():.2f} px / FDE {err[:, -1].mean():.2f} px "
          f"/ 分类 {(logits.argmax(-1) == labels).mean():.4f}  (应与测试成绩一致)")

    lines = [f"# 误差分解分析报告 [{TAG}]\n",
             f"- 模型: {os.path.basename(CKPT_PATH)}  测试集: seed={TEST_SEED} x {B} 条\n",
             f"- 总体: ADE {err.mean():.2f} px, FDE {err[:, -1].mean():.2f} px, "
             f"分类 {(logits.argmax(-1) == labels).mean():.4f}\n"]

    # ---------- 1. 反弹帧 vs 平稳帧 ----------
    bounce = bounce_mask_future(centers)               # (B,F,N)
    eb, es = err[bounce], err[~bounce]
    ratio = eb.mean() / es.mean()
    print(f"\n[1] 反弹帧误差 {eb.mean():.2f} px (占 {bounce.mean()*100:.1f}% 帧)  "
          f"vs 平稳帧 {es.mean():.2f} px  -> 反弹/平稳 = {ratio:.2f}x")
    lines += ["\n## 1. 撞墙反弹 vs 平稳帧\n",
              f"- 反弹帧: {eb.mean():.2f} px ({bounce.mean()*100:.1f}% 帧), "
              f"平稳帧: {es.mean():.2f} px, 比值 {ratio:.2f}x\n"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(1, F + 1)
    ax.plot(x, err.mean(axis=(0, 2)), "-o", color="gray", label="all frames")
    ax.plot(x, [err[:, f][bounce[:, f]].mean() if bounce[:, f].any() else np.nan
                for f in range(F)], "--^", color="orange", label="bounce frames")
    ax.plot(x, [err[:, f][~bounce[:, f]].mean() for f in range(F)], "-s", color="green",
            label="non-bounce frames")
    ax.set_xlabel("future frame")
    ax.set_ylabel("mean error (px)")
    ax.set_title("Error on bounce vs non-bounce frames")
    ax.set_xticks(x)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(os.path.join(FIG_DIR, f"analysis_bounce_{TAG}.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)

    # ---------- 2. 误差分布 + 槽位交换 ----------
    seq_ade = err.mean(axis=(1, 2))                    # (B,) 每序列 ADE
    p = np.percentile(seq_ade, [50, 75, 90, 99])
    order = np.argsort(seq_ade)[::-1]
    total = seq_ade.sum()
    top1 = seq_ade[order[:B // 100]].sum() / total * 100
    top5 = seq_ade[order[:B // 20]].sum() / total * 100
    top10 = seq_ade[order[:B // 10]].sum() / total * 100
    print(f"\n[2] 每序列 ADE 分位数: p50 {p[0]:.2f} / p75 {p[1]:.2f} / p90 {p[2]:.2f} / "
          f"p99 {p[3]:.2f} / max {seq_ade.max():.2f} px")
    print(f"    误差贡献集中度: top1% 样本贡献 {top1:.1f}%, top5% 贡献 {top5:.1f}%, "
          f"top10% 贡献 {top10:.1f}% (均匀时应为 1/5/10%)")
    lines += ["\n## 2. 误差分布与集中度\n",
              f"- 每序列 ADE 分位数: p50 {p[0]:.2f} / p75 {p[1]:.2f} / p90 {p[2]:.2f} / "
              f"p99 {p[3]:.2f} / max {seq_ade.max():.2f} px\n",
              f"- 误差贡献: top1% 样本占 {top1:.1f}%, top5% 占 {top5:.1f}%, top10% 占 {top10:.1f}%\n"]

    # 槽位交换检测: 预测离"另一个数字"的真值比离自己的更近
    own = err                                          # (B,F,N) 到自己真值的距离
    other = np.stack([np.linalg.norm(pred[:, :, n] - target[:, :, 1 - n], axis=-1)
                      for n in range(N)], axis=-1)     # (B,F,N) 到另一个数字真值的距离
    swap_frames = (other < own).sum(axis=1)            # (B,N)
    swapped = (swap_frames >= SWAP_FRAME_TH).any(axis=1)   # (B,)
    n_swap = swapped.sum()
    ade_keep = seq_ade[~swapped].mean()
    print(f"    槽位交换序列: {n_swap}/{B} ({n_swap / B * 100:.1f}%), "
          f"这些序列平均 ADE {seq_ade[swapped].mean() if n_swap else 0:.2f} px; "
          f"剔除后总体 ADE {ade_keep:.2f} px")
    lines += [f"- 槽位交换序列 (>= {SWAP_FRAME_TH} 帧预测离另一数字更近): "
              f"{n_swap}/{B} ({n_swap / B * 100:.1f}%), 其平均 ADE "
              f"{seq_ade[swapped].mean() if n_swap else 0:.2f} px; 剔除后总体 ADE {ade_keep:.2f} px\n"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(seq_ade, bins=60, color="steelblue", edgecolor="white")
    for v, name in zip(p, ["p50", "p75", "p90", "p99"]):
        ax.axvline(v, ls="--", lw=1, label=f"{name} = {v:.1f}")
    ax.set_xlabel("per-sequence ADE (px)")
    ax.set_ylabel("count")
    ax.set_title("Distribution of per-sequence ADE")
    ax.legend()
    fig.savefig(os.path.join(FIG_DIR, f"analysis_error_dist_{TAG}.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)

    # ---------- 3. 分类错误 vs 数字接近程度 ----------
    cls_ok = (logits.argmax(-1) == labels).all(axis=1)         # (B,) 两数字全对
    digit_ok = (logits.argmax(-1) == labels)                   # (B,N) 逐数字
    min_dist = min_center_dist_input(centers)
    print(f"\n[3] 分类按输入窗口最小质心距离分桶 (逐数字准确率 | 两数字全对率):")
    lines += ["\n## 3. 分类错误 vs 数字接近程度\n",
              "| 最小质心距离 (px) | 序列占比 | 逐数字准确率 | 两数字全对率 |\n",
              "|---|---|---|---|\n"]
    for lo, hi in [(0, 14), (14, 20), (20, 28), (28, 1e9)]:
        m = (min_dist >= lo) & (min_dist < hi)
        if not m.any():
            continue
        name = f"{lo}~{hi if hi < 1e8 else '∞'}"
        print(f"    距离 {name:8s}: {m.mean()*100:5.1f}% 序列  "
              f"digit_acc {digit_ok[m].mean():.4f}  both_ok {cls_ok[m].mean():.4f}")
        lines.append(f"| {name} | {m.mean()*100:.1f}% | {digit_ok[m].mean():.4f} "
                     f"| {cls_ok[m].mean():.4f} |\n")

    # 槽位交换与接近程度的交叉: 交换是否都发生在近距离接触时?
    close = min_dist < 14
    print(f"\n    交叉验证: 槽位交换序列中 { (close & swapped).sum() / max(n_swap,1) *100:.1f}% "
          f"曾发生深度重叠(最小距离<14px); 深度重叠序列占全体 {close.mean()*100:.1f}%")
    lines.append(f"- 槽位交换序列中 {(close & swapped).sum() / max(n_swap,1) *100:.1f}% "
                 f"曾深度重叠(最小距离<14px); 深度重叠序列占全体 {close.mean()*100:.1f}%\n")

    # ---------- 误差最大的 6 条样本可视化 ----------
    fig, axes = plt.subplots(6, 3, figsize=(13, 4 * 6))
    for row, i in enumerate(order[:6]):
        video = ds[i]["video"].numpy()
        c, pd_ = centers[i], pred[i]
        base = None
        for j, t in enumerate([N_INPUT - 1, 14, 19]):
            ax = axes[row, j]
            ax.imshow(video[t, 0], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(f"seq {i} (ADE {seq_ade[i]:.1f}px) frame {t}"
                         + ("  [SWAP]" if swapped[i] else ""), fontsize=10)
            ax.axis("off")
            draw_tracks(ax, c, pd_, base)
        axes[row, 0].set_ylabel(f"true {labels[i].tolist()} pred {logits[i].argmax(-1).tolist()}",
                                fontsize=10)
    fig.suptitle("Top-6 worst sequences: observed(white) true(green) model(red)")
    fig.savefig(os.path.join(FIG_DIR, f"analysis_worst_{TAG}.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)

    report_path = os.path.join(OUT_DIR, f"误差分析_{TAG}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"\n图已保存到 {FIG_DIR} (analysis_worst_{TAG}.png 等)")
    print("文字报告 ->", report_path)


if __name__ == "__main__":
    main()
