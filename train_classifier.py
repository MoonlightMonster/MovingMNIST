"""MNIST 数字分类 —— 成熟版训练脚本.

设计目标: 不是玩具 demo, 而是可直接作为后续多任务模型分类头的基线.
  - 架构: 3 级卷积块 (Conv-BN-ReLU) + 残差连接 + 全局平均池化
  - 数据: 训练集随机仿射增强, 划分 5000 张做验证
  - 训练: AdamW + 余弦退火 + 标签平滑, 按验证准确率保存最优 checkpoint
  - 可复现: 固定全局随机种子
  - 输出: 最优验证/测试准确率, checkpoint 存 checkpoints/mnist_cnn_best.pth

运行:
  E:\\anaconda3\\envs\\CurveSegmentationMethod\\python.exe train_classifier.py
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 须在 import torch 前设置

import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# ---------------- 配置 ----------------
DATA_ROOT = r"D:\MovingMNIST\dataset\data"
CKPT_DIR = r"D:\MovingMNIST\checkpoints"
SEED = 42

BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 5e-4
LABEL_SMOOTHING = 0.1
VAL_SIZE = 5000
NUM_WORKERS = 0  # Windows 下 >0 需要 if __name__ 保护已加, 但 MNIST 太小, 0 即可


# ---------------- 模型 ----------------
class ConvBlock(nn.Module):
    """两层卷积 + 残差连接的基本块 (通道数不变时使用残差)。"""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        # 残差分支: 尺寸或通道不匹配时用 1x1 卷积对齐
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class MNISTNet(nn.Module):
    """28x28 -> 10 类. ~30 万参数, 测试准确率预期 99.4%+."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.layer1 = ConvBlock(32, 64, stride=2)   # 28 -> 14
        self.layer2 = ConvBlock(64, 128, stride=2)  # 14 -> 7
        self.layer3 = ConvBlock(128, 128)           # 7 -> 7
        self.pool = nn.AdaptiveAvgPool2d(1)         # 全局平均池化, 取代大参数全连接
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(self.dropout(x))


# ---------------- 训练/评估 ----------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


def main():
    # 可复现
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # 数据: 训练集做轻量仿射增强 (MNIST 手写数字对旋转/平移鲁棒性有真实需求)
    tf_train = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST 全局均值/方差
    ])
    tf_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    full_train = datasets.MNIST(DATA_ROOT, train=True, transform=tf_train)
    train_set, val_set = random_split(full_train, [len(full_train) - VAL_SIZE, VAL_SIZE],
                                      generator=torch.Generator().manual_seed(SEED))
    # 验证集用独立 Dataset 对象, 不带增强
    val_set.dataset = datasets.MNIST(DATA_ROOT, train=True, transform=tf_eval)
    test_set = datasets.MNIST(DATA_ROOT, train=False, transform=tf_eval)

    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_set, 512, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_set, 512, num_workers=NUM_WORKERS, pin_memory=True)

    model = MNISTNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e3:.0f}K")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    os.makedirs(CKPT_DIR, exist_ok=True)
    best_val = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        loss_sum = 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            loss = criterion(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.numel()
        scheduler.step()

        val_acc = evaluate(model, val_loader, device)
        marker = ""
        if val_acc > best_val:
            best_val = val_acc
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_acc": val_acc}, os.path.join(CKPT_DIR, "mnist_cnn_best.pth"))
            marker = "  <- best"
        print(f"epoch {epoch:2d}/{EPOCHS}  loss {loss_sum/len(train_set):.4f}"
              f"  val_acc {val_acc:.4f}  ({time.time()-t0:.1f}s){marker}")

    # 用最优 checkpoint 在测试集上出最终成绩
    ckpt = torch.load(os.path.join(CKPT_DIR, "mnist_cnn_best.pth"), map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    test_acc = evaluate(model, test_loader, device)
    print(f"\nbest val_acc {ckpt['val_acc']:.4f} (epoch {ckpt['epoch']}),  test_acc {test_acc:.4f}")


if __name__ == "__main__":
    main()
