"""步骤 3 v2: ConvLSTM 多任务模型改进版 + 消融开关 —— 针对 v1 的两个瓶颈.

v1 (train_convlstm.py) 首训结果: ADE 7.81 / FDE 9.98 / 分类 95.6%,
诊断: ①分类卡在 ~96% 是结构问题——全局池化丢失空间绑定, 重叠时两数字身份分不清;
     ②轨迹"向均值收缩"是 MSE 损失在不确定时预测条件均值的典型行为.

v2 两处候选改进 (打包同训结果为负: ADE 13.06 / 分类 93.9%, 锚定被分类梯度带偏),
故做成可独立开关的消融实验, 逐个验证:
  --cls-mode slot : 分类头按槽位取样 (用锚定位置在 h1 上 grid_sample 局部特征 +
                    全局特征, 分别分类); 配合 --detach-anchor 切断分类梯度对锚定的污染
  --traj-mode delta + --traj-loss huber : 轨迹改预测位移量累加 (贴合匀速+反弹结构),
                    损失换 Smooth L1 (鼓励中位数而非均值, 减轻收缩)

对照组约定 (其余配置均与 v1 相同):
  v1  = --cls-mode gap  --traj-mode abs   --traj-loss mse   (即原 train_convlstm.py)
  v2a = --cls-mode gap  --traj-mode delta --traj-loss huber (只改轨迹)
  v2b = --cls-mode slot --traj-mode abs   --traj-loss mse --detach-anchor (只改分类)

运行:
  E:\\anaconda3\\envs\\CurveSegmentationMethod\\python.exe train_convlstm_v2.py --tag v2a --cls-mode gap --traj-mode delta --traj-loss huber
  E:\\anaconda3\\envs\\CurveSegmentationMethod\\python.exe train_convlstm_v2.py --tag v2b --cls-mode slot --traj-mode abs --traj-loss mse --detach-anchor
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

# ---------------- 默认配置 (与 v1 一致) ----------------
DATA_ROOT = r"D:\MovingMNIST\dataset\data"
CKPT_DIR = r"D:\MovingMNIST\checkpoints"
SAMPLE_DIR = r"D:\MovingMNIST\samples"
SEED = 42

N_INPUT = 10      # 观测帧数
N_FUTURE = 10     # 预测帧数
CANVAS = 64       # 画布边长 (用于坐标归一化与指标换算)
N_SLOTS = 2       # 每条序列的数字个数

L_TRAJ = 1.0      # 未来轨迹损失权重
L_ANCHOR = 0.5    # 当前帧锚定损失权重
L_VEL = 0.5       # 速度估计辅助损失权重 (仅 cls_mode=clean)
HUBER_BETA = 0.02  # Smooth L1 转折点 (归一化坐标), 0.02*64 ≈ 1.3 px
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
    """逐帧编码: CNN 提空间特征, 两层 ConvLSTM 建模时间与运动. (与 v1 相同)"""

    def __init__(self, h1_ch=96, h2_ch=128):
        super().__init__()
        self.stem = nn.Sequential(  # 64 -> 32 -> 16
            nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.clstm1 = ConvLSTMCell(64, h1_ch)     # @16x16
        self.down = nn.Sequential(                # 16 -> 8
            nn.Conv2d(h1_ch, h2_ch, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(h2_ch), nn.ReLU(),
        )
        self.clstm2 = ConvLSTMCell(h2_ch, h2_ch)  # @8x8

    def forward(self, video):
        """video: (B,T,1,64,64) -> 最终隐状态 h1 (B,C1,16,16), h2 (B,C2,8,8),
        h1_seq (B,T,C1,16,16) 每帧特征 (v5 分类头按时间融合用)."""
        B = video.shape[0]
        h1, c1 = self.clstm1.init_state(B, (16, 16), video.device, video.dtype)
        h2, c2 = self.clstm2.init_state(B, (8, 8), video.device, video.dtype)
        h1_seq = []
        for t in range(video.shape[1]):
            x = self.stem(video[:, t])
            h1, c1 = self.clstm1(x, (h1, c1))
            h2, c2 = self.clstm2(self.down(h1), (h2, c2))
            h1_seq.append(h1)
        return h1, h2, torch.stack(h1_seq, dim=1)


class MultiTaskConvLSTMv2(nn.Module):
    """共享编码器 + 可切换的分类头 (gap/slot) 与轨迹解码 (abs/delta).

    slot_feat=True (v4): 解码器每步用各槽位当前预测位置 (detach 防梯度污染)
    在 h1 上 grid_sample 该数字的局部外观特征, 与位置一起输入 LSTM —
    用"外观+运动"双线索维持槽位身份, 针对误差分析定位的根因 (深度重叠时身份交换)."""

    def __init__(self, n_slots=N_SLOTS, n_classes=10, n_future=N_FUTURE, dec_hidden=512,
                 h1_ch=96, h2_ch=128, cls_mode="slot", traj_mode="delta", detach_anchor=False,
                 slot_feat=False, slot_feat_dim=32, slot_anchor=False, anchor_dim=32):
        super().__init__()
        assert cls_mode in ("gap", "slot", "clean") and traj_mode in ("abs", "delta")
        self.n_slots, self.n_future = n_slots, n_future
        self.cls_mode, self.traj_mode, self.detach_anchor = cls_mode, traj_mode, detach_anchor
        self.slot_feat = slot_feat
        self.slot_anchor = slot_anchor
        self.encoder = SpatioTemporalEncoder(h1_ch, h2_ch)

        # 轨迹解码器
        self.ctx = nn.Linear(h2_ch * 8 * 8, dec_hidden)
        self.h0 = nn.Linear(dec_hidden, dec_hidden)
        self.c0 = nn.Linear(dec_hidden, dec_hidden)
        if slot_feat:  # 每槽位: h1 取样特征 -> slot_feat_dim 维
            self.slot_proj = nn.Sequential(nn.Linear(h1_ch, slot_feat_dim), nn.ReLU())
        if slot_anchor:  # 身份锚: 观测段每槽位"最清晰帧"外观 -> anchor_dim 维 (detach 防污染)
            self.anchor_proj = nn.Sequential(nn.Linear(h1_ch, anchor_dim), nn.ReLU())
        cell_in = n_slots * 2
        cell_in += (n_slots * slot_feat_dim if slot_feat else 0)
        cell_in += (n_slots * anchor_dim if slot_anchor else 0)
        self.cell = nn.LSTMCell(cell_in, dec_hidden)
        self.pos_out = nn.Sequential(
            nn.Linear(dec_hidden, 256), nn.ReLU(),
            nn.Linear(256, n_slots * 2),
        )

        # 分类头: gap = v1 全局池化; slot = 锚定点取样; clean = v5 沿时间维融合
        # (锚定 + 速度估计反推过去每帧位置, 按帧间距离加权, 聚焦最干净的时刻)
        if cls_mode == "clean":
            self.vel_out = nn.Linear(dec_hidden, n_slots * 2)   # 瞬时速度估计 (辅助监督)
            self.tau = nn.Parameter(torch.tensor(0.1))          # 时间融合温度 (可学习)
            self.cls_head = nn.Sequential(
                nn.Linear(h1_ch + h2_ch, 128), nn.ReLU(), nn.Dropout(0.25),
                nn.Linear(128, n_classes),
            )
        elif cls_mode == "gap":
            self.cls_head = nn.Sequential(
                nn.Linear(h2_ch, 128), nn.ReLU(), nn.Dropout(0.25),
                nn.Linear(128, n_slots * n_classes),
            )
        else:
            self.cls_head = nn.Sequential(
                nn.Linear(h1_ch + h2_ch, 128), nn.ReLU(), nn.Dropout(0.25),
                nn.Linear(128, n_classes),
            )

    def forward(self, video, obs_centers=None):
        """video: (B,T,1,64,64) -> logits (B,N,10), traj (B,F+1,N,2), vel (B,N,2)|None.
        traj[:, 0] 是当前帧锚定位置, traj[:, 1:] 是未来 F 帧预测 (均为归一化坐标).
        vel 仅 cls_mode=clean 时输出: 锚定时刻的瞬时速度估计 (归一化坐标/帧).
        obs_centers: (B,T,N,2)|None 观测段每帧每数字质心(归一化, 推理时来自输入视频质心,
        训练时来自真值), 仅 slot_anchor 需要, 用于在"最清晰帧"取样身份外观锚."""
        h1, h2, h1_seq = self.encoder(video)
        B = video.shape[0]

        # 身份锚: 观测段每槽位在"最清晰帧"(两数字间距最大的帧)取样外观, detach 防梯度污染.
        id_feat = None
        if self.slot_anchor and obs_centers is not None:
            with torch.no_grad():
                oc = obs_centers.detach()                       # (B,T,N,2) 归一化坐标
                # 逐帧两数字间距, 选最清晰帧 (间距最大, 最不重叠)
                dist = (oc[:, :, 0] - oc[:, :, 1]).norm(dim=-1)  # (B,T)
                best_t = dist.argmax(dim=1)                      # (B,)
                b_idx = torch.arange(B, device=video.device)
                anchor_pos = oc[b_idx, best_t]                   # (B,N,2) 最清晰帧位置
                grid = (anchor_pos * 2 - 1).view(B, self.n_slots, 1, 2)
                local = F.grid_sample(h1, grid, mode="bilinear",
                                      padding_mode="border", align_corners=False)
                local = local.squeeze(-1).permute(0, 2, 1)       # (B,N,C1)
                id_feat = self.anchor_proj(local).flatten(1)     # (B,N*anchor_dim)

        ctx = torch.tanh(self.ctx(h2.flatten(1)))
        h, c = torch.tanh(self.h0(ctx)), torch.tanh(self.c0(ctx))
        pos = video.new_zeros(B, self.n_slots * 2)
        preds = []
        vel = None
        for t in range(self.n_future + 1):
            feat = None
            if self.slot_feat and t > 0:
                # v4: 在各槽位当前预测位置取样外观特征 (坐标 detach, 防梯度污染)
                grid = (pos.detach() * 2 - 1).view(B, self.n_slots, 1, 2)
                local = F.grid_sample(h1, grid, mode="bilinear",
                                      padding_mode="border", align_corners=False)
                feat = self.slot_proj(local.squeeze(-1).permute(0, 2, 1)).flatten(1)
            elif self.slot_feat:
                feat = video.new_zeros(B, self.n_slots * 32)  # 首步无位置, 零特征
            parts = [pos]
            if feat is not None:
                parts.append(feat)
            if id_feat is not None:
                parts.append(id_feat)
            inp = torch.cat(parts, dim=1)
            h, c = self.cell(inp, (h, c))
            if t == 0 and self.cls_mode == "clean":
                vel = self.vel_out(h).view(B, self.n_slots, 2)  # v5: 瞬时速度估计
            out = self.pos_out(h)
            if t == 0 or self.traj_mode == "abs":
                pos = out            # 首步恒为绝对锚定; abs 模式每步都出绝对坐标
            else:
                pos = pos + out      # delta 模式: 之后每步预测位移累加
            preds.append(pos)
        traj = torch.stack(preds, dim=1).view(B, self.n_future + 1, self.n_slots, 2)

        if self.cls_mode == "gap":
            logits = self.cls_head(h2.mean(dim=(2, 3))).view(B, self.n_slots, -1)
        elif self.cls_mode == "slot":
            anchor = traj[:, 0].detach() if self.detach_anchor else traj[:, 0]
            grid = (anchor * 2 - 1).view(B, self.n_slots, 1, 2)   # grid_sample 坐标系 [-1,1]
            local = F.grid_sample(h1, grid, mode="bilinear",
                                  padding_mode="border", align_corners=False)  # (B,C1,N,1)
            local = local.squeeze(-1).permute(0, 2, 1)             # (B,N,C1)
            glob = h2.mean(dim=(2, 3)).unsqueeze(1).expand(-1, self.n_slots, -1)
            logits = self.cls_head(torch.cat([local, glob], dim=-1))  # (B,N,10)
        else:  # clean: v5 沿时间维融合最干净帧的槽位特征
            anchor = traj[:, 0]                                     # (B,N,2)
            steps = (N_INPUT - 1 - torch.arange(N_INPUT, device=video.device,
                                                dtype=traj.dtype)).view(1, N_INPUT, 1, 1)
            pos_in = (anchor.unsqueeze(1) - steps * vel.unsqueeze(1)).detach()  # (B,T,N,2) 反推
            dist = (pos_in[:, :, 0] - pos_in[:, :, 1]).norm(dim=-1)   # (B,T) 帧间距离
            w = torch.softmax(dist / self.tau.abs().clamp(min=0.01), dim=1)  # 干净帧权重
            maps = h1_seq.flatten(0, 1)                               # (B*T,C1,16,16)
            grid = (pos_in * 2 - 1).view(B * N_INPUT, self.n_slots, 1, 2)
            feat = F.grid_sample(maps, grid, mode="bilinear",
                                 padding_mode="border", align_corners=False)  # (B*T,C1,N,1)
            feat = feat.squeeze(-1).permute(0, 2, 1).view(B, N_INPUT, self.n_slots, -1)
            local = (w.view(B, N_INPUT, 1, 1) * feat).sum(dim=1)      # (B,N,C1) 时间融合
            glob = h2.mean(dim=(2, 3)).unsqueeze(1).expand(-1, self.n_slots, -1)
            logits = self.cls_head(torch.cat([local, glob], dim=-1))  # (B,N,10)
        return logits, traj, vel


# ---------------- 目标构造与损失 ----------------
def _load_weights_with_migration(model, state_dict):
    """加载 checkpoint 权重, 兼容结构扩展 (新增 anchor_proj / cell 输入维扩大).

    - 缺失的层 (如 anchor_proj) 保持随机初始化.
    - cell.weight_ih 形状扩大时: 旧权重复制到前部对应列, 新增列随机初始化.
    - 其余 shape 匹配的层用 strict=False 加载.
    """
    new_sd = model.state_dict()
    # 1) 逐键迁移: 形状匹配直接复制; cell 输入维扩大则部分复制
    for k, v in state_dict.items():
        if k not in new_sd:
            continue
        if new_sd[k].shape == v.shape:
            new_sd[k] = v
        elif "cell.weight_ih" in k and new_sd[k].shape[1] > v.shape[1]:
            # 输入维扩大: 旧列复制到前部, 新增列保持随机初始化
            new_sd[k][:, :v.shape[1]] = v
        # 其它形状不匹配的键跳过 (保留随机初始化)
    model.load_state_dict(new_sd, strict=False)


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


def multitask_loss(logits, traj, labels, centers_norm, traj_loss, vel=None, cls_weight=1.0):
    """logits (B,N,10), traj (B,F+1,N,2), labels (B,N), centers_norm (B,T,N,2),
    vel (B,N,2)|None: clean 分类头的瞬时速度估计, 用真值末帧速度做辅助监督."""
    loss_cls = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               labels.reshape(-1), label_smoothing=LABEL_SMOOTHING)
    if traj_loss == "huber":
        loss_anchor = F.smooth_l1_loss(traj[:, 0], centers_norm[:, N_INPUT - 1], beta=HUBER_BETA)
        loss_traj = F.smooth_l1_loss(traj[:, 1:], centers_norm[:, N_INPUT:], beta=HUBER_BETA)
    else:
        loss_anchor = F.mse_loss(traj[:, 0], centers_norm[:, N_INPUT - 1])
        loss_traj = F.mse_loss(traj[:, 1:], centers_norm[:, N_INPUT:])
    total = cls_weight * loss_cls + L_TRAJ * loss_traj + L_ANCHOR * loss_anchor
    loss_vel = logits.new_zeros(())
    if vel is not None:
        vel_gt = centers_norm[:, N_INPUT - 1] - centers_norm[:, N_INPUT - 2]  # 末帧瞬时速度
        loss_vel = F.smooth_l1_loss(vel, vel_gt, beta=HUBER_BETA)
        total = total + L_VEL * loss_vel
    return total, loss_cls.detach(), loss_traj.detach(), loss_anchor.detach(), loss_vel.detach()


# ---------------- 评估 (与 v1 同口径) ----------------
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
            obs_centers = (centers[:, :N_INPUT] / CANVAS) if model.slot_anchor else None
            logits, traj, _ = model(video[:, :N_INPUT], obs_centers)

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
            obs_centers = (centers[:, :N_INPUT] / CANVAS) if model.slot_anchor else None
            logits, traj, _ = model(video[:, :N_INPUT], obs_centers)
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
    fig.suptitle("ConvLSTM v2: observed (white) / true (green) / predicted (red)")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------- 主流程 ----------------
def main():
    parser = argparse.ArgumentParser(description="ConvLSTM 多任务训练 v2 (消融实验版)")
    parser.add_argument("--tag", default="v2", help="输出文件标识 (checkpoint/metrics/图)")
    parser.add_argument("--cls-mode", choices=["gap", "slot", "clean"], default="slot",
                        help="gap=v1全局池化头; slot=按锚定位置取样; clean=v5沿时间维融合最干净帧")
    parser.add_argument("--traj-mode", choices=["abs", "delta"], default="delta",
                        help="abs=每步直接出绝对坐标(v1); delta=首步锚定+位移累加")
    parser.add_argument("--traj-loss", choices=["mse", "huber"], default="huber")
    parser.add_argument("--detach-anchor", action="store_true",
                        help="slot 分类头取样时切断梯度, 防止分类损失带偏锚定位置")
    parser.add_argument("--slot-feat", action="store_true",
                        help="v4: 解码器每步喂槽位级外观特征 (grid_sample h1, 坐标detach)")
    parser.add_argument("--slot-anchor", action="store_true",
                        help="v5: 解码器喂观测段'最清晰帧'的身份外观锚 (detach, 维持重叠时身份)")
    parser.add_argument("--h1-ch", type=int, default=96, help="ConvLSTM 第1层通道数")
    parser.add_argument("--h2-ch", type=int, default=128, help="ConvLSTM 第2层通道数")
    parser.add_argument("--dec-hidden", type=int, default=512, help="轨迹解码器隐层宽度")
    parser.add_argument("--select", choices=["ade", "acc"], default="ade",
                        help="最优 checkpoint 选取标准: ade=最小验证ADE; acc=最高验证分类")
    parser.add_argument("--cls-weight", type=float, default=1.0, help="分类损失权重 (默认1, 微调可调高)")
    parser.add_argument("--init-from", default=None, help="从指定 checkpoint 加载权重继续训练 (微调)")
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

    ckpt_path = os.path.join(CKPT_DIR, f"convlstm_{args.tag}_best.pth")
    metrics_path = os.path.join(CKPT_DIR, f"convlstm_{args.tag}_metrics.json")
    fig_path = os.path.join(SAMPLE_DIR, f"traj_convlstm_{args.tag}.png")
    print(f"[{args.tag}] cls_mode={args.cls_mode} traj_mode={args.traj_mode} "
          f"traj_loss={args.traj_loss} detach_anchor={args.detach_anchor} slot_feat={args.slot_feat} "
          f"slot_anchor={args.slot_anchor}")

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

    model = MultiTaskConvLSTMv2(cls_mode=args.cls_mode, traj_mode=args.traj_mode,
                                detach_anchor=args.detach_anchor, h1_ch=args.h1_ch,
                                h2_ch=args.h2_ch, dec_hidden=args.dec_hidden,
                                slot_feat=args.slot_feat, slot_anchor=args.slot_anchor).to(device)
    if args.init_from:
        init = torch.load(args.init_from, map_location=device, weights_only=True)
        _load_weights_with_migration(model, init["model_state_dict"])
        print(f"已从 {args.init_from} 加载权重 (epoch {init['epoch']}, "
              f"val_ade {init['val_ade']:.2f}px) 继续训练")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(CKPT_DIR, exist_ok=True)
    best_score = float("inf")  # score 统一"越小越好": ade 模式为 val_ade, acc 模式为 -val_acc
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        sums = np.zeros(5)  # total, cls, traj, anchor, vel
        for batch in train_loader:
            video = batch["video"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            centers = batch["centers"].to(device, non_blocking=True)
            labels, centers = sort_slots(labels, centers)
            centers_norm = centers / CANVAS

            with torch.amp.autocast("cuda", enabled=use_amp):
                obs_centers = centers_norm[:, :N_INPUT] if args.slot_anchor else None
                logits, traj, vel = model(video[:, :N_INPUT], obs_centers)
                loss, l_cls, l_traj, l_anchor, l_vel = multitask_loss(
                    logits, traj, labels, centers_norm, args.traj_loss, vel,
                    cls_weight=args.cls_weight)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            sums += [loss.item(), l_cls.item(), l_traj.item(), l_anchor.item(), l_vel.item()]
        scheduler.step()

        n_batch = len(train_loader)
        val_ade, val_fde, val_acc = evaluate(model, val_loader, device, use_amp)
        score = val_ade if args.select == "ade" else -val_acc
        marker = ""
        if score < best_score:
            best_score = score
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_ade": val_ade, "val_fde": val_fde, "val_acc": val_acc,
                        "config": vars(args)}, ckpt_path)
            marker = "  <- best"
        print(f"epoch {epoch:2d}/{args.epochs}  loss {sums[0]/n_batch:.4f} "
              f"(cls {sums[1]/n_batch:.4f} traj {sums[2]/n_batch:.4f} anchor {sums[3]/n_batch:.4f} "
              f"vel {sums[4]/n_batch:.4f})  "
              f"val_ade {val_ade:.2f}px  val_fde {val_fde:.2f}px  val_acc {val_acc:.4f}  "
              f"({time.time()-t0:.1f}s){marker}")

    # ---- 用最优 checkpoint 在固定测试集上出最终成绩 (与 v1/线性基线同 seed 同条数) ----
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    test_ade, test_fde, test_acc = evaluate(model, test_loader, device, use_amp)

    print(f"\n=== [{args.tag}] 最终结果 (best epoch {ckpt['epoch']}, 按 val ADE 选取) ===")
    print(f"分类准确率 (逐数字): {test_acc:.4f}   [v1: 0.9560, 达标线 >= 0.99]")
    print(f"轨迹 ADE: {test_ade:.2f} px   [v1: 7.81, 线性基线 23.88, 理想 < 5]")
    print(f"轨迹 FDE: {test_fde:.2f} px   [v1: 9.98, 线性基线 41.60]")

    metrics = {"tag": args.tag, "epoch": ckpt["epoch"], "test_ade": test_ade,
               "test_fde": test_fde, "test_acc": test_acc, "val_ade": ckpt["val_ade"],
               "config": vars(args)}
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    visualize(model, test_set, device, fig_path)
    print("可视化已保存到", fig_path)


if __name__ == "__main__":
    main()
