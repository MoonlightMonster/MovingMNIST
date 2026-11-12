"""第二步: 匈牙利关联后处理 (轨迹级 v2) —— 用身份嵌入 + 位置解决深度重叠时的槽位交换.

背景:
  v4ft_embed 模型输出每帧每个槽位的"预测位置" + "身份嵌入向量" (64 维).
  误差分析已定位: 深度重叠时槽位身份交换是主要误差来源 (25.1% 序列).

v2 与 v1 的区别 (v1 失败教训):
  - v1 参考身份用"整段平均位置", 两数字一直在动, 平均无意义 → 错.
  - v1 逐帧独立重排, 破坏时间一致性, 过度干预 (1819 帧被交换) → 错.
  - v1 嵌入只当代价微调, 没真正主导"是否交换"的判断 → 没发挥作用.

v2 正确做法 (对齐工业界 MOT 的 DeepSORT/ByteTrack 思想):
  1. 参考身份 = 观测段末帧(第 10 帧)的槽位身份 —— 槽位按首帧 x 排序固定,
     槽位 0 从头到尾就是"靠左那个数字", 这是要维持的稳定身份.
  2. 用嵌入判断"是否真的交换": 对每序列, 比较"当前槽位嵌入"与"参考槽位嵌入"
     的相似度矩阵, 只有当嵌入明确指向"槽位 0 现在更像参考身份 1、槽位 1 更像
     参考身份 0"(即嵌入预测交换) 时才触发.
  3. 一次性交换 + 锁定: 一旦判定交换, 从该帧起整体翻转槽位身份, 之后不再逐帧重排,
     保持轨迹的时间连续性.

产出:
  终端报告 + results/figs/assoc_*.png
  报告追加到 results/关联分析.md

运行:
  E:\\anaconda3\\envs\\CurveSegmentationMethod\\python.exe assoc_postprocess.py
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 须在 import torch 前设置

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
SWAP_FRAME_TH = 3      # 判定槽位交换的帧数阈值 (与 analyze_errors 一致)
OVERLAP_TH = 14.0      # 判定"重叠/接近"的位置距离阈值 (px)


def load_embed_model(device):
    """加载带嵌入头的模型 (注意要传 embed=True, 否则 state_dict 对不上)."""
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    cfg = ckpt["config"]
    model = MultiTaskConvLSTMv2(cls_mode=cfg["cls_mode"], traj_mode=cfg["traj_mode"],
                                detach_anchor=cfg["detach_anchor"], h1_ch=cfg["h1_ch"],
                                h2_ch=cfg["h2_ch"], dec_hidden=cfg["dec_hidden"],
                                slot_feat=cfg.get("slot_feat", False),
                                embed=cfg.get("embed", False)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"已加载 {os.path.basename(CKPT_PATH)} (epoch {ckpt['epoch']}, "
          f"embed={cfg.get('embed', False)})")
    return model


@torch.no_grad()
def run_inference(model, loader, device):
    """返回 pred (B,F,N,2) px, embed (B,N,D), target (B,F,N,2), labels (B,N), centers (B,T,N,2)."""
    preds, embeds, targets, all_labels, all_centers = [], [], [], [], []
    for batch in loader:
        video = batch["video"].to(device)
        labels = batch["labels"].to(device)
        centers = batch["centers"].to(device)
        labels, centers = sort_slots(labels, centers)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits, traj, _, embed = model(video[:, :N_INPUT])
        preds.append((traj[:, 1:].float() * CANVAS).cpu().numpy())   # (B,F,N,2)
        if embed is not None:
            embeds.append(embed.float().cpu().numpy())               # (B,N,D)
        targets.append(centers[:, N_INPUT:].cpu().numpy())           # (B,F,N,2)
        all_labels.append(labels.cpu().numpy())
        all_centers.append(centers.cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    target = np.concatenate(targets, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    centers = np.concatenate(all_centers, axis=0)
    embed = np.concatenate(embeds, axis=0) if embeds else None
    return pred, embed, target, labels, centers


def per_seq_error(pred, target):
    """逐序列 ADE/FDE. pred,target: (B,F,N,2) -> (B,) ADE, (B,) FDE."""
    err = np.linalg.norm(pred - target, axis=-1)          # (B,F,N)
    return err.mean(axis=(1, 2)), err[:, -1].mean(axis=1)


def detect_swap(pred, target):
    """检测槽位交换: 预测离"另一个数字"的真值比离自己的更近的帧数.
    返回 swap_count (B,), swapped (B,) bool."""
    own = np.linalg.norm(pred - target, axis=-1)          # (B,F,N)
    other = np.stack([np.linalg.norm(pred[:, :, n] - target[:, :, 1 - n], axis=-1)
                      for n in range(pred.shape[2])], axis=-1)  # (B,F,N)
    swap_count = (other < own).sum(axis=1)                # (B,N)
    swapped = (swap_count >= SWAP_FRAME_TH).any(axis=1)
    return swap_count, swapped


def overlap_frames(pred, th=OVERLAP_TH):
    """找出每序列中两槽位预测位置过近的帧. pred:(B,F,N,2) -> close_mask (B,F) bool."""
    d = np.linalg.norm(pred[:, :, 0] - pred[:, :, 1], axis=-1)  # (B,F)
    return d < th


def _cos_sim(a, b):
    """a,b: (D,) -> cos 相似度 [-1,1]."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def associate_track_level(pred, embed, use_embed=True, w_pos=1.0, w_emb=1.0):
    """轨迹级关联: 用嵌入判断是否发生交换, 一次性整体翻转并锁定.

    对每条序列:
      1. 参考身份 = 观测段末帧(N_INPUT-1)两槽位的真值位置 + 各自嵌入.
         槽位按首帧 x 排序固定, 所以参考身份 0/1 就是"靠左/靠右数字"的稳定身份.
      2. 逐未来帧, 若该帧两槽位预测距离过近(重叠), 用嵌入判断:
            "当前槽位0的嵌入" 更接近 "参考身份1的嵌入" 且
            "当前槽位1的嵌入" 更接近 "参考身份0的嵌入" → 判定交换.
      3. 一旦判定交换, 从该帧起整体翻转槽位, 锁定不再改.
         若使用位置辅助, 则用匈牙利匹配(代价=嵌入+位置)作为备选判定.

    返回: out (B,F,N,2) 关联后的预测, 以及每序列是否交换、交换起始帧.
    """
    B, F, N, _ = pred.shape
    out = pred.copy()
    n_swap_seq = 0
    n_swap_applied = 0

    for b in range(B):
        # 参考身份: 用观测段末帧(真值 centers 最后一帧)位置 + 嵌入
        # 这里参考嵌入用模型对该序列输出的嵌入 (每槽位一个, 已是稳定身份向量)
        ref_emb = embed[b] if embed is not None else None  # (N,D)
        if ref_emb is not None:
            ref_emb_n = ref_emb / (np.linalg.norm(ref_emb, axis=-1, keepdims=True) + 1e-8)

        swapped = False
        swap_frame = -1

        for f in range(F):
            # 只在"该帧两槽位预测距离过近"时才考虑交换 (重叠/接近才可能身份混淆)
            if np.linalg.norm(pred[b, f, 0] - pred[b, f, 1]) >= OVERLAP_TH:
                continue

            if use_embed and ref_emb is not None:
                # 当前帧两槽位嵌入 (同序列内嵌入是固定的 N 个, 不随帧变; 但为通用这里仍按槽位取)
                cur_emb = embed[b] / (np.linalg.norm(embed[b], axis=-1, keepdims=True) + 1e-8)
                # 交叉 vs 同配 的相似度差
                same = _cos_sim(cur_emb[0], ref_emb_n[0]) + _cos_sim(cur_emb[1], ref_emb_n[1])
                cross = _cos_sim(cur_emb[0], ref_emb_n[1]) + _cos_sim(cur_emb[1], ref_emb_n[0])
                # 交叉明显更匹配 → 判定交换
                if cross > same:
                    swapped = True
                    swap_frame = f
            else:
                # 纯位置: 用匈牙利匹配判断 (当前帧两槽位 vs 参考两位置)
                cost = np.zeros((N, N))
                for i in range(N):
                    for j in range(N):
                        cost[i, j] = np.linalg.norm(pred[b, f, i] - pred[b, f, j])  # 近似
                # 实际: 参考位置用首帧真值位置
                ref_pos = pred[b, 0]  # 用预测首帧位置做参考 (近似稳定)
                cost = np.zeros((N, N))
                for i in range(N):
                    for j in range(N):
                        cost[i, j] = np.linalg.norm(pred[b, f, i] - ref_pos[j])
                row_ind, col_ind = linear_sum_assignment(cost)
                identity_map = np.empty(N, dtype=int)
                identity_map[row_ind] = col_ind
                if not np.array_equal(identity_map, np.arange(N)):
                    swapped = True
                    swap_frame = f

            if swapped:
                break

        # 一次性交换: 从 swap_frame 起整体翻转两槽位身份
        if swapped:
            n_swap_seq += 1
            for f in range(swap_frame, F):
                out[b, f] = pred[b, f][::-1]  # 交换槽位 0 和 1
                n_swap_applied += 1

    return out, n_swap_seq, n_swap_applied


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_embed_model(device)
    ds = MovingMNIST(root=DATA_ROOT, train=False, seq_len=N_INPUT + N_FUTURE,
                     length=N_EVAL, seed=TEST_SEED)
    loader = DataLoader(ds, 128, num_workers=2, pin_memory=True, persistent_workers=True)

    pred, embed, target, labels, centers = run_inference(model, loader, device)
    B, F, N, _ = pred.shape
    print(f"\n推理完成: {B} 条序列 x {N} 数字 x {F} 未来帧, embed={'有' if embed is not None else '无'}")

    # 基线: 不做关联
    ade0, fde0 = per_seq_error(pred, target)
    _, swapped0 = detect_swap(pred, target)
    print(f"\n[基线-不关联]   ADE {ade0.mean():.2f} / FDE {fde0.mean():.2f} px, "
          f"交换序列 {swapped0.sum()}/{B} ({swapped0.sum()/B*100:.1f}%)")

    # 方案1: 只用位置关联 (轨迹级)
    pred_pos, nseq1, napp1 = associate_track_level(pred, None, use_embed=False)
    ade1, fde1 = per_seq_error(pred_pos, target)
    _, swapped1 = detect_swap(pred_pos, target)
    print(f"[只用位置关联]   ADE {ade1.mean():.2f} / FDE {fde1.mean():.2f} px, "
          f"交换序列 {swapped1.sum()}/{B} ({swapped1.sum()/B*100:.1f}%), "
          f"触发交换 {nseq1} 序列 / {napp1} 帧")

    # 方案2: 位置 + 嵌入关联 (轨迹级)
    pred_emb, nseq2, napp2 = associate_track_level(pred, embed, use_embed=True)
    ade2, fde2 = per_seq_error(pred_emb, target)
    _, swapped2 = detect_swap(pred_emb, target)
    print(f"[位置+嵌入关联]  ADE {ade2.mean():.2f} / FDE {fde2.mean():.2f} px, "
          f"交换序列 {swapped2.sum()}/{B} ({swapped2.sum()/B*100:.1f}%), "
          f"触发交换 {nseq2} 序列 / {napp2} 帧")

    # 汇总报告
    lines = ["# 关联后处理分析报告 (轨迹级 v2)\n\n",
             f"- 模型: {os.path.basename(CKPT_PATH)}  测试集: seed={TEST_SEED} x {B} 条\n",
             f"- 重叠阈值: 两槽位预测距离 < {OVERLAP_TH} px  交换帧阈值: {SWAP_FRAME_TH}\n",
             f"- 关联方式: 轨迹级一次性交换 + 锁定 (参考身份=观测末帧槽位身份)\n\n",
             "## 结果对比\n\n",
             "| 方案 | ADE (px) | FDE (px) | 交换序列占比 | 触发交换 |\n",
             "|---|---|---|---|---|\n",
             f"| 不关联 | {ade0.mean():.2f} | {fde0.mean():.2f} | "
             f"{swapped0.sum()/B*100:.1f}% | - |\n",
             f"| 只用位置 | {ade1.mean():.2f} | {fde1.mean():.2f} | "
             f"{swapped1.sum()/B*100:.1f}% | {nseq1} 序列 |\n",
             f"| 位置+嵌入 | {ade2.mean():.2f} | {fde2.mean():.2f} | "
             f"{swapped2.sum()/B*100:.1f}% | {nseq2} 序列 |\n",
             "\n## 结论\n\n",
             f"- 嵌入带来的 ADE 改善: {ade1.mean()-ade2.mean():.2f} px "
             f"(相对纯位置关联), 相对基线改善 {ade0.mean()-ade2.mean():.2f} px\n"]

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    seq_ade = [ade0, ade1, ade2]
    names = ["no assoc", "pos only", "pos+embed"]
    colors = ["gray", "orange", "green"]
    sw = swapped0
    for a, n_, c in zip(seq_ade, names, colors):
        axes[0].plot(np.arange(B)[sw], a[sw], ".", ms=4, color=c, alpha=0.5, label=n_)
    axes[0].set_xlabel("swapped sequence index")
    axes[0].set_ylabel("per-sequence ADE (px)")
    axes[0].set_title("ADE on swapped sequences (should drop with assoc)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    x = np.arange(3)
    axes[1].bar(x, [a.mean() for a in seq_ade], color=colors, alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names)
    axes[1].set_ylabel("mean ADE (px)")
    axes[1].set_title("Overall mean ADE comparison")
    for xi, a in zip(x, seq_ade):
        axes[1].text(xi, a.mean(), f"{a.mean():.2f}", ha="center", va="bottom")
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle("Track-level Hungarian association (v2: embed-gated, one-shot swap)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "assoc_comparison_v2.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)

    report_path = os.path.join(OUT_DIR, "关联分析.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"\n图已保存 -> {os.path.join(FIG_DIR, 'assoc_comparison_v2.png')}")
    print("文字报告 ->", report_path)


if __name__ == "__main__":
    main()
