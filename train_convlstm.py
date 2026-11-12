  SegmentationMethod\\python.exe train_convlstm.py
  快速自检 (几分钟跑通全流程):
  E:\\anaconda3\\envs\\CurveSegmentationMethod\\python.exe train_convlstm.py --smoke
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 该环境存在双 libiomp5md.dll, 须在 import torch 前设置

import argparse
import json
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moving_mnist import MovingMNIST

# ---------------- 默认配置 ----------------
DATA_ROOT = r"D:\MovingMNIST\dataset\data"
CKPT_DIR = r"D:\MovingMNIST\checkpoints"
SAMPLE_DIR = r"D:\MovingMNIST\samples"
CKPT_PATH = os.path.join(CKPT_DIR, "convlstm_multitask_best.pth")
SEED = 42

N_INPUT = 10      # 观测帧数
N_FUTURE = 10     # 预测帧数
CANVAS = 64       # 画布边长 (用于坐标归一化与指标换算)
N_SLOTS = 2       # 每条序列的数字个数

L_TRAJ = 1.0      # 未来轨迹损失权重
L_ANCHOR = 0.5    # 当前帧锚定损失权重
LABEL_SMOOTHING = 0.1
GRAD_CLIP = 1.0   # LSTM 训练必备, 防梯度爆炸

VAL_SEED = 555    # 验证集固定 seed (调模型用)
TEST_SEED = 2026  # 测试集固定 seed (与线性基线同一套, 直接对比)


# ---------------- 模型 ----------------
class ConvLSTMCell(nn.Module):
    """标准 ConvLSTM 单元 (Shi et al. 2015): 门控用卷积而非全连接, 保留空间结构."""

    def __init__(self, in_ch, hidden_ch, kernel=3):
        super().__init__()
        self.hidden_ch = hidden_ch
        self.conv = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch, kernel, padding=kernel // 2)

    def forward(self, x, state):
        h, c = state
        i, f, g, o = self.conv(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        c_new = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_new = torch.sigmoid(o) * torch.tanh(c_new)
        return h_new, c_new

    def init_state(self, batch, spatial, device, dtype):
        h, w = spatial
        return (torch.zeros(batch, self.hidden_ch, h, w, device=device, dtype=dtype),
                torch.zeros(batch, self.hidden_ch, h, w, device=device, dtype=dtype))


class SpatioTemporalEncoder(nn.Module):
    """逐帧编码: CNN 提空间特征, 两层 ConvLSTM 建模时间与运动."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(  # 64 -> 32 -> 16
            nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.clstm1 = ConvLSTMCell(64, 96)    # @16x16
        self.down = nn.Sequential(            # 16 -> 8
            nn.Conv2d(96, 128, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.clstm2 = ConvLSTMCell(128, 128)  # @8x8

    def forward(self, video):
        """video: (B,T,1,64,64) -> 最终隐状态 h1 (B,96,16,16), h2 (B,128,8,8)."""
        B = video.shape[0]
        h1, c1 = self.clstm1.init_state(B, (16, 16), video.device, video.dtype)
        h2, c2 = self.clstm2.init_state(B, (8, 8), video.device, video.dtype)
        for t in range(video.shape[1]):
            x = self.stem(video[:, t])
            h1, c1 = self.clstm1(x, (h1, c1))
            h2, c2 = self.clstm2(self.down(h1), (h2, c2))
        return h1, h2


class MultiTaskConvLSTM(nn.Module):
    """共享编码器 + 分类头 + 自回归轨迹解码器."""

    def __init__(self, n_slots=N_SLOTS, n_classes=10, n_future=N_FUTURE, dec_hidden=512):
        super().__init__()
        self.n_slots, self.n_future = n_slots, n_future
        self.encoder = SpatioTemporalEncoder()

        self.cls_head = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(128, n_slots * n_classes),
        )

        # 轨迹解码器: 初始状态由编码器最终隐状态 (保留空间信息的 flatten) 变换而来
        self.ctx = nn.Linear(128 * 8 * 8, dec_hidden)
        self.h0 = nn.Linear(dec_hidden, dec_hidden)
        self.c0 = nn.Linear(dec_hidden, dec_hidden)
        self.cell = nn.LSTMCell(n_slots * 2, dec_hidden)
        self.pos_out = nn.Sequential(
            nn.Linear(dec_hidden, 256), nn.ReLU(),
            nn.Linear(256, n_slots * 2),
        )

    def forward(self, video):
        """video: (B,T,1,64,64) -> logits (B,N,10), traj (B,F+1,N,2).
        traj[:, 0] 是当前帧锚定位置, traj[:, 1:] 是未来 F 帧预测 (均为归一化坐标)."""
        _, h2 = self.encoder(video)
        B = video.shape[0]

        logits = self.cls_head(h2.mean(dim=(2, 3))).view(B, self.n_slots, -1)

        ctx = torch.tanh(self.ctx(h2.flatten(1)))
        h, c = torch.tanh(self.h0(ctx)), torch.tanh(self.c0(ctx))
        pos = video.new_zeros(B, self.n_slots * 2)
        preds = []
        for _ in range(self.n_future + 1):
            h, c = self.cell(pos, (h, c))
            pos = self.pos_out(h)          # 归一化绝对坐标
            preds.append(pos)
        traj = torch.stack(preds, dim=1).view(B, self.n_future + 1, self.n_slots, 2)
        return logits, traj


# ---------------- 目标构造与损失 ----------------
def sort_slots(labels, centers):
    """按首帧质心 x(并列看 y) 排序数字槽位, 使输出顺序有确定含义.
    labels: (B,N), centers: (B,T,N,2) -> 排序后的 labels, centers."""
    key = centers[:, 0, :, 0] * 1000.0 + centers[:, 0, :, 1]
    order = key.argsort(dim=1)                                   # (B,N)
    labels = torch.take_along_dim(labels, order, dim=1)
    T = centers.shape[1]
    order_exp = order.unsqueeze(1).unsqueeze(-1).expand(-1, T, -1, 2)
    centers = torch.take_along_dim(centers, order_exp, dim=2)
    return labels, centers


def multitask_loss(logits, traj, labels, centers_norm):
    """logits (B,N,10), traj (B,F+1,N,2), labels (B,N), centers_norm (B,T,N,2)."""
    loss_cls = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               labels.reshape(-1), label_smoothing=LABEL_SMOOTHING)
    loss_anchor = F.mse_loss(traj[:, 0], centers_norm[:, N_INPUT - 1])
    loss_traj = F.mse_loss(traj[:, 1:], centers_norm[:, N_INPUT:])
    total = loss_cls + L_TRAJ * loss_traj + L_ANCHOR * loss_anchor
    return total, loss_cls.detach(), loss_traj.detach(), loss_anchor.detach()


# ---------------- 评估 ----------------
@torch.no_grad()
def evaluate(model, loader, device, use_amp):
    """返回 ADE(px), FDE(px), 逐数字分类准确率 —— 与线性基线同口径."""
    model.eval()
    n_seq = 0
    ade_sum = fde_sum = 0.0
    correct = 0
    for batch in loader:
        video = batch["video"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        centers = batch["centers"].to(device, non_blocking=True)
        labels, centers = sort_slots(labels, centers)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits, traj = model(video[:, :N_INPUT])

        err = (traj[:, 1:].float() * CANVAS - centers[:, N_INPUT:]).norm(dim=-1)  # (B,F,N) 像素
        ade_sum += err.mean(dim=(1, 2)).sum().item()   # 每条序列先平均, 与基线同口径
        fde_sum += err[:, -1].mean(dim=1).sum().item()
        correct += (logits.argmax(-1) == labels).sum().item()
        n_seq += video.shape[0]

    return ade_sum / n_seq, fde_sum / n_seq, correct / (n_seq * N_SLOTS)


# ---------------- 可视化 ----------------
@torch.no_grad()
def visualize(model, ds, device, path, n_samples=3):
    """观测轨迹(白) / 真实未来(绿) / 模型预测(红), 与基线图同风格便于对比."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    for i in range(n_samples):
        s = ds[i]
        video = s["video"].unsqueeze(0).to(device)          # (1,T,1,H,W)
        labels = s["labels"].unsqueeze(0).to(device)
        centers = s["centers"].unsqueeze(0).to(device)
        labels, centers = sort_slots(labels, centers)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits, traj = model(video[:, :N_INPUT])
        pred = (traj[0, 1:].float() * CANVAS).cpu().numpy()     # (F,N,2)
        pred_cls = logits[0].argmax(-1).cpu().tolist()
        true_cls = labels[0].cpu().tolist()
        centers = centers[0].cpu().numpy()

        for j, t in enumerate([0, N_INPUT - 1, N_INPUT + N_FUTURE - 1]):
            ax = axes[i, j]
            ax.imshow(video[0, t, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"sample {i} frame {t}", fontsize=10)
            ax.axis("off")
            for n, marker in zip(range(N_SLOTS), ["o", "s"]):
                ax.plot(centers[:N_INPUT, n, 0], centers[:N_INPUT, n, 1], "-" + marker,
                        color="white", ms=3, lw=1, label="observed" if (i == 0 and n == 0) else None)
                ax.plot(centers[N_INPUT:, n, 0], centers[N_INPUT:, n, 1], "-" + marker,
                        color="lime", ms=3, lw=1, label="true future" if (i == 0 and n == 0) else None)
                ax.plot(pred[:, n, 0], pred[:, n, 1], "--x",
                        color="red", ms=3, lw=1, label="model pred" if (i == 0 and n == 0) else None)
        axes[i, 0].set_ylabel(f"true {true_cls} / pred {pred_cls}", fontsize=10)
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle("ConvLSTM multi-task: observed (white) / true (green) / predicted (red)")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------- 主流程 ----------------
def main():
    parser = argparse.ArgumentParser(description="ConvLSTM 多任务训练 (分类 + 轨迹预测)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-len", type=int, default=12800, help="每个 epoch 在线生成的序列数")
    parser.add_argument("--val-len", type=int, default=1000)
    parser.add_argument("--test-len", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true", help="关闭混合精度")
    parser.add_argument("--smoke", action="store_true", help="小规模自检 (跑通全流程即可)")
    args = parser.parse_args()

    if args.smoke:
        args.epochs, args.train_len, args.val_len, args.test_len = 2, 512, 128, 128
        args.batch_size, args.num_workers = 32, 0
        print("[smoke] 小规模自检模式")

    # 可复现
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and not args.no_amp
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
          + f"  amp: {use_amp}")

    # 数据: 训练 seed=None 在线随机生成 (每个 epoch 数据都不同); 验证/测试固定 seed
    train_set = MovingMNIST(root=DATA_ROOT, train=True, seq_len=N_INPUT + N_FUTURE,
                            length=args.train_len, seed=None)
    val_set = MovingMNIST(root=DATA_ROOT, train=True, seq_len=N_INPUT + N_FUTURE,
                          length=args.val_len, seed=VAL_SEED)
    test_set = MovingMNIST(root=DATA_ROOT, train=False, seq_len=N_INPUT + N_FUTURE,
                           length=args.test_len, seed=TEST_SEED)

    train_loader = DataLoader(train_set, args.batch_size, shuffle=False,  # 数据本身随机, 无需 shuffle
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=args.num_workers > 0)
    eval_workers = min(2, args.num_workers)
    val_loader = DataLoader(val_set, 128, num_workers=eval_workers, pin_memory=True,
                            persistent_workers=eval_workers > 0)
    test_loader = DataLoader(test_set, 128, num_workers=eval_workers, pin_memory=True,
                             persistent_workers=eval_workers > 0)

    model = MultiTaskConvLSTM().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(CKPT_DIR, exist_ok=True)
    best_ade = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        sums = np.zeros(4)  # total, cls, traj, anchor
        for batch in train_loader:
            video = batch["video"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            centers = batch["centers"].to(device, non_blocking=True)
            labels, centers = sort_slots(labels, centers)
            centers_norm = centers / CANVAS

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, traj = model(video[:, :N_INPUT])
                loss, l_cls, l_traj, l_anchor = multitask_loss(logits, traj, labels, centers_norm)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            sums += [loss.item(), l_cls.item(), l_traj.item(), l_anchor.item()]
        scheduler.step()

        n_batch = len(train_loader)
        val_ade, val_fde, val_acc = evaluate(model, val_loader, device, use_amp)
        marker = ""
        if val_ade < best_ade:  # 轨迹是难任务, 以验证 ADE 选最优
            best_ade = val_ade
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_ade": val_ade, "val_fde": val_fde, "val_acc": val_acc,
                        "config": vars(args)}, CKPT_PATH)
            marker = "  <- best"
        print(f"epoch {epoch:2d}/{args.epochs}  loss {sums[0]/n_batch:.4f} "
              f"(cls {sums[1]/n_batch:.4f} traj {sums[2]/n_batch:.4f} anchor {sums[3]/n_batch:.4f})  "
              f"val_ade {val_ade:.2f}px  val_fde {val_fde:.2f}px  val_acc {val_acc:.4f}  "
              f"({time.time()-t0:.1f}s){marker}")

    # ---- 用最优 checkpoint 在固定测试集上出最终成绩 (与线性基线同 seed 同条数) ----
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    test_ade, test_fde, test_acc = evaluate(model, test_loader, device, use_amp)

    print(f"\n=== 最终结果 (best epoch {ckpt['epoch']}, 按 val ADE 选取) ===")
    print(f"分类准确率 (逐数字): {test_acc:.4f}   [达标线 >= 0.99]")
    print(f"轨迹 ADE: {test_ade:.2f} px   [线性基线 23.88, 理想 < 5]")
    print(f"轨迹 FDE: {test_fde:.2f} px   [线性基线 41.60]")

    metrics = {"epoch": ckpt["epoch"], "test_ade": test_ade, "test_fde": test_fde,
               "test_acc": test_acc, "val_ade": ckpt["val_ade"], "config": vars(args)}
    with open(os.path.join(CKPT_DIR, "convlstm_multitask_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    fig_path = os.path.join(SAMPLE_DIR, "traj_convlstm.png")
    visualize(model, test_set, device, fig_path)
    print("可视化已保存到", fig_path)


if __name__ == "__main__":
    main()
