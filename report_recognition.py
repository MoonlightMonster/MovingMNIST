"""汇报可视化: 端到端识别过程图 (输入 -> 分类 -> 轨迹预测).

产出 (results/figs/):
  report_recognition.png  — 一个样本的完整识别过程:
     左: 输入10帧(灰度)  中: 分类结果(类别+置信度条形)  右: 轨迹预测(观测/真值/预测叠加)
  report_recognition_multi.png — 4个样本并列, 展示不同难度场景(重叠/反弹/交换)

运行: E:\\anaconda3\\envs\\CurveSegmentationMethod\\python.exe report_recognition.py
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


def predict_one(model, device, sample):
    """对单个样本做完整前向, 返回所有中间结果."""
    video = sample["video"].unsqueeze(0).to(device)
    labels = sample["labels"].unsqueeze(0).to(device)
    centers = sample["centers"].unsqueeze(0).to(device)
    labels, centers = sort_slots(labels, centers)
    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        obs_centers = centers[:, :N_INPUT] / CANVAS
        logits, traj, vel, heatmap = model(video[:, :N_INPUT], obs_centers)
    prob = torch.softmax(logits, dim=-1)          # (1,N,10)
    pred_cls = prob.argmax(-1)                     # (1,N)
    conf = prob.gather(-1, pred_cls.unsqueeze(-1)).squeeze(-1)  # (1,N)
    return {
        "video": video[0].cpu().numpy(),           # (10,1,64,64)
        "centers": centers[0].cpu().numpy(),       # (20,N,2)
        "true_cls": labels[0].cpu().tolist(),
        "pred_cls": pred_cls[0].detach().cpu().tolist(),
        "conf": conf[0].detach().cpu().tolist(),
        "prob": prob[0].detach().cpu().numpy(),             # (N,10) 完整概率分布
        "traj": (traj[0, 1:].detach().float() * CANVAS).cpu().numpy(),  # (F,N,2)
    }


def draw_recognition_panel(ax_input, ax_cls, ax_traj, r, sample_idx):
    """画单个样本的三栏识别过程."""
    video = r["video"]       # (10,1,64,64)
    centers = r["centers"]   # (20,N,2)
    traj = r["traj"]         # (F,N,2)
    true_cls = r["true_cls"]
    pred_cls = r["pred_cls"]
    conf = r["conf"]
    prob = r["prob"]

    # --- 左: 输入帧 (取连续6帧 t=0..5, 2行3列网格排列) ---
    frames = [0, 1, 2, 3, 4, 5]
    ax_input.axis("off")
    # 拼接成 2行x3列 大图
    row1 = np.concatenate([video[f, 0] for f in frames[:3]], axis=1)
    row2 = np.concatenate([video[f, 0] for f in frames[3:]], axis=1)
    strip = np.concatenate([row1, row2], axis=0)
    ax_input.imshow(strip, cmap="gray", vmin=0, vmax=1)
    # 标注帧号 (放在帧内左上角)
    for i, f in enumerate(frames):
        r, c = divmod(i, 3)
        ax_input.text(c * 64 + 3, r * 64 + 12, f"t={f}", ha="left", va="center",
                      fontsize=11, color="#ff4444", fontweight="bold")
    ax_input.set_title("输入：前10帧观测视频（展示前6帧）", fontsize=12)

    # --- 中: 分类结果 (每个数字的类别+置信度条形) ---
    ax_cls.axis("off")
    digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for slot in range(2):
        axb = ax_cls.inset_axes([0.08 + slot * 0.5, 0.15, 0.4, 0.75])
        p = prob[slot]
        colors = ["#c0c0c0"] * 10
        colors[pred_cls[slot]] = "#d43a3a" if pred_cls[slot] == true_cls[slot] else "#2e8b57"
        axb.barh(digits, p * 100, color=colors, height=0.7)
        axb.set_yticks(digits)
        axb.set_yticklabels([str(d) for d in digits], fontsize=8)
        axb.set_xlim(0, 100)
        axb.set_xlabel("置信度 %", fontsize=8)
        ok = "正确" if pred_cls[slot] == true_cls[slot] else "错误"
        axb.set_title(f"数字{slot+1}: 预测 {pred_cls[slot]}（{ok}）\n"
                      f"(真值 {true_cls[slot]}, 置信 {conf[slot]*100:.0f}%)",
                      fontsize=10)
    ax_cls.set_title("分类识别：逐数字类别 + 置信度", fontsize=11)

    # --- 右: 轨迹预测 (叠加在末帧观测上) ---
    ax_traj.imshow(video[N_INPUT - 1, 0], cmap="gray", vmin=0, vmax=1)
    for n, (mk, name) in enumerate(zip(["o", "s"], ["数字1", "数字2"])):
        obs = centers[:N_INPUT, n]
        fut = centers[N_INPUT:, n]
        p = traj[:, n]
        ax_traj.plot(obs[:, 0], obs[:, 1], "-" + mk, color="white", ms=4, lw=2,
                     label="观测" if n == 0 else None)
        ax_traj.plot(fut[:, 0], fut[:, 1], "-" + mk, color="lime", ms=4, lw=2,
                     label="真值" if n == 0 else None)
        ax_traj.plot(p[:, 0], p[:, 1], "--x", color="red", ms=5, lw=2,
                     label="预测" if n == 0 else None)
    ax_traj.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax_traj.set_title("轨迹预测：未来10帧质心（白观测/绿真值/红预测）", fontsize=11)
    ax_traj.axis("off")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    ds = MovingMNIST(root=DATA_ROOT, train=False, seq_len=N_INPUT + N_FUTURE,
                     length=100, seed=TEST_SEED)

    # 挑样本: 覆盖不同场景 (重叠/反弹/交换)
    samples = [3, 7, 15, 22]
    results = [predict_one(model, device, ds[i]) for i in samples]

    # 图A: 单个样本详细三栏
    fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                             gridspec_kw={"width_ratios": [1.6, 1.3, 1.0]})
    draw_recognition_panel(axes[0], axes[1], axes[2], results[0], samples[0])
    fig.suptitle("模型端到端识别过程（输入 → 分类识别 → 轨迹预测）",
                 fontsize=16, y=1.0)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "report_recognition.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("图A 已保存 ->", p)

    # 图B: 4样本并列 (只画轨迹预测, 展示不同难度)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    for i, (r, idx) in enumerate(zip(results, samples)):
        video = r["video"]; centers = r["centers"]; traj = r["traj"]
        ax = axes[i]
        ax.imshow(video[N_INPUT - 1, 0], cmap="gray", vmin=0, vmax=1)
        for n, mk in enumerate(["o", "s"]):
            ax.plot(centers[:N_INPUT, n, 0], centers[:N_INPUT, n, 1], "-" + mk,
                    color="white", ms=4, lw=2)
            ax.plot(centers[N_INPUT:, n, 0], centers[N_INPUT:, n, 1], "-" + mk,
                    color="lime", ms=4, lw=2)
            ax.plot(traj[:, n, 0], traj[:, n, 1], "--x", color="red", ms=5, lw=2)
        ok = "对" if r["pred_cls"] == r["true_cls"] else "错"
        ax.set_title(f"样本{idx}  预测{r['pred_cls']} 真值{r['true_cls']}({ok})",
                     fontsize=12)
        ax.axis("off")
    axes[0].legend(["观测", "真值", "预测"], loc="upper right",
                   fontsize=10, framealpha=0.9)
    fig.suptitle("多样本识别结果（白=观测 / 绿=真值 / 红=模型预测）", fontsize=15, y=1.0)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "report_recognition_multi.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("图B 已保存 ->", p)

    print("\n所有识别过程图已生成。")


if __name__ == "__main__":
    main()
