"""成果可视化: 当前最优模型 (v3) vs 线性基线.

产出 (都在 D:\\MovingMNIST\\samples\\):
  v3_vs_baseline.png  静态对比大图: 4 条样本 x 3 个关键帧, 同图叠加
                      观测(白) / 真实未来(绿) / 模型预测(红) / 线性基线(蓝)
  v3_error_curve.png  逐帧误差曲线: 模型 vs 基线的平均欧氏误差 (500 条测试序列)
  v3_sample_*.gif     动画: 视频逐帧播放, 轨迹实时叠加

运行:
  E:\\anaconda3\\envs\\CurveSegmentationMethod\\python.exe visualize_results.py
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 须在 import torch 前设置

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
CKPT_PATH = os.environ.get("MM_CKPT", r"D:\MovingMNIST\checkpoints\convlstm_v3_best.pth")
OUT_DIR = r"D:\MovingMNIST\samples"
TEST_SEED = 2026
N_ERROR_EVAL = 500   # 误差曲线用的序列数 (与基线同口径的子集)
N_SHOW = 4           # 静态图/GIF 展示的样本数

C_OBS, C_TRUE, C_PRED, C_BASE = "white", "lime", "red", "deepskyblue"


def linear_extrapolate(obs, n_future):
    """obs: (n_input, 2) -> (n_future, 2) 匀速外推, 与步骤2基线一致."""
    v = (obs[-1] - obs[0]) / (len(obs) - 1)
    k = np.arange(1, n_future + 1, dtype=np.float64)[:, None]
    return obs[-1] + k * v


def load_model(device):
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    cfg = ckpt["config"]
    model = MultiTaskConvLSTMv2(cls_mode=cfg["cls_mode"], traj_mode=cfg["traj_mode"],
                                detach_anchor=cfg["detach_anchor"], h1_ch=cfg["h1_ch"],
                                h2_ch=cfg["h2_ch"], dec_hidden=cfg["dec_hidden"],
                                slot_feat=cfg.get("slot_feat", False)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"已加载 {CKPT_PATH} (epoch {ckpt['epoch']}, val_ade {ckpt['val_ade']:.2f}px)")
    return model


@torch.no_grad()
def predict(model, video, device):
    """video: (1,T,1,H,W) -> pred (F,N,2) px, logits (1,N,10)."""
    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        logits, traj, _ = model(video[:, :N_INPUT].to(device))
    pred = (traj[0, 1:].float() * CANVAS).cpu().numpy()
    return pred, logits.cpu()


def draw_tracks(ax, centers, pred, base, show_legend=False):
    """在一帧画面上叠加轨迹. centers: (T,N,2) px, pred/base: (F,N,2) px."""
    for n, marker in zip(range(centers.shape[1]), ["o", "s"]):
        obs, future = centers[:N_INPUT, n], centers[N_INPUT:, n]
        kw = dict(ms=3, lw=1)
        ax.plot(obs[:, 0], obs[:, 1], "-" + marker, color=C_OBS,
                label="observed" if show_legend and n == 0 else None, **kw)
        ax.plot(future[:, 0], future[:, 1], "-" + marker, color=C_TRUE,
                label="true future" if show_legend and n == 0 else None, **kw)
        ax.plot(pred[:, n, 0], pred[:, n, 1], "--x", color=C_PRED,
                label="model pred" if show_legend and n == 0 else None, **kw)
        if base is not None:
            ax.plot(base[:, n, 0], base[:, n, 1], "--^", color=C_BASE,
                    label="linear baseline" if show_legend and n == 0 else None, **kw)


def make_static_grid(model, ds, device):
    """4 条样本 x 3 帧 (第9/14/19帧) 的模型 vs 基线对比大图."""
    fig, axes = plt.subplots(N_SHOW, 3, figsize=(13, 4 * N_SHOW))
    for i in range(N_SHOW):
        s = ds[i]
        video_np = s["video"].numpy()
        labels = s["labels"].unsqueeze(0).to(device)
        centers = s["centers"].unsqueeze(0).to(device)
        labels, centers = sort_slots(labels, centers)
        centers = centers[0].cpu().numpy()

        pred, logits = predict(model, s["video"].unsqueeze(0), device)
        pred_cls = logits[0].argmax(-1).tolist()
        true_cls = labels[0].tolist()
        base = np.stack([linear_extrapolate(centers[:N_INPUT, n], N_FUTURE)
                         for n in range(centers.shape[1])], axis=1)  # (F,N,2)

        for j, t in enumerate([N_INPUT - 1, 14, 19]):
            ax = axes[i, j]
            ax.imshow(video_np[t, 0], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(f"sample {i}  frame {t}", fontsize=10)
            ax.axis("off")
            draw_tracks(ax, centers, pred, base, show_legend=(i == 0 and j == 0))
        ok = "OK" if pred_cls == true_cls else "X"
        axes[i, 0].set_ylabel(f"cls true {true_cls} pred {pred_cls} [{ok}]", fontsize=10)
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle("v3 model vs linear baseline: observed(white) true(green) model(red) baseline(blue)")
    path = os.path.join(OUT_DIR, "v3_vs_baseline.png")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("静态对比图 ->", path)


def make_error_curve(model, device):
    """500 条固定测试序列上, 模型与基线的逐帧平均误差曲线."""
    ds = MovingMNIST(root=DATA_ROOT, train=False, seq_len=N_INPUT + N_FUTURE,
                     length=N_ERROR_EVAL, seed=TEST_SEED)
    loader = DataLoader(ds, 128, num_workers=2, pin_memory=True,
                        persistent_workers=True)
    model_err = torch.zeros(N_FUTURE)
    base_err = torch.zeros(N_FUTURE)
    n_seq = 0
    for batch in loader:
        video = batch["video"].to(device)
        labels = batch["labels"].to(device)
        centers = batch["centers"].to(device)
        labels, centers = sort_slots(labels, centers)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            _, traj = model(video[:, :N_INPUT])
        pred = traj[:, 1:].float() * CANVAS                  # (B,F,N,2)
        target = centers[:, N_INPUT:]
        model_err += (pred - target).norm(dim=-1).mean(dim=2).sum(dim=0).cpu()  # (F,) 按序列求和

        obs = centers[:, :N_INPUT].cpu().numpy().astype(np.float64)  # (B,I,N,2)
        v = (obs[:, -1] - obs[:, 0]) / (N_INPUT - 1)                 # (B,N,2)
        k = np.arange(1, N_FUTURE + 1)[None, :, None, None]
        base_pred = obs[:, -1:] + k * v[:, None]                     # (B,F,N,2)
        base_err += torch.from_numpy(
            np.linalg.norm(base_pred - target.cpu().numpy(), axis=-1).mean(axis=2).sum(axis=0))
        n_seq += video.shape[0]

    model_err /= n_seq
    base_err /= n_seq
    x = np.arange(1, N_FUTURE + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, base_err.numpy(), "--^", color=C_BASE, lw=2,
            label=f"linear baseline  (ADE {base_err.mean():.2f} px)")
    ax.plot(x, model_err.numpy(), "-x", color=C_PRED, lw=2,
            label=f"v3 model  (ADE {model_err.mean():.2f} px)")
    ax.set_xlabel("future frame")
    ax.set_ylabel("mean euclidean error (px)")
    ax.set_title(f"Per-frame prediction error on {n_seq} test sequences (seed={TEST_SEED})")
    ax.set_xticks(x)
    ax.grid(alpha=0.3)
    ax.legend()
    path = os.path.join(OUT_DIR, "v3_error_curve.png")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"误差曲线 -> {path}  (模型 ADE {model_err.mean():.2f} vs 基线 {base_err.mean():.2f})")


def make_gifs(model, ds, device, fps=5):
    """每条样本一个 gif: 视频逐帧播放 + 轨迹叠加."""
    from PIL import Image

    for i in range(N_SHOW):
        s = ds[i]
        video_np = s["video"].numpy()
        labels = s["labels"].unsqueeze(0).to(device)
        centers = s["centers"].unsqueeze(0).to(device)
        labels, centers = sort_slots(labels, centers)
        centers = centers[0].cpu().numpy()
        pred, logits = predict(model, s["video"].unsqueeze(0), device)
        pred_cls = logits[0].argmax(-1).tolist()
        true_cls = labels[0].tolist()

        imgs = []
        for t in range(N_INPUT + N_FUTURE):
            fig, ax = plt.subplots(figsize=(3.2, 3.6), dpi=100)
            ax.imshow(video_np[t, 0], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(f"frame {t}  |  true {true_cls} pred {pred_cls}", fontsize=9)
            ax.axis("off")
            draw_tracks(ax, centers, pred, base=None, show_legend=(t == 0))
            if t == 0:
                ax.legend(loc="upper right", fontsize=7)
            fig.tight_layout()
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())
            imgs.append(Image.fromarray(buf[:, :, :3]))
            plt.close(fig)

        path = os.path.join(OUT_DIR, f"v3_sample_{i}.gif")
        imgs[0].save(path, save_all=True, append_images=imgs[1:],
                     duration=1000 // fps, loop=0)
        print("GIF ->", path)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    ds = MovingMNIST(root=DATA_ROOT, train=False, seq_len=N_INPUT + N_FUTURE,
                     length=N_SHOW, seed=TEST_SEED)  # 与基线图同一批样本
    os.makedirs(OUT_DIR, exist_ok=True)
    make_static_grid(model, ds, device)
    make_error_curve(model, device)
    make_gifs(model, ds, device)
    print("\n全部完成, 输出目录:", OUT_DIR)


if __name__ == "__main__":
    main()
