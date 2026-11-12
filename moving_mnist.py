"""Moving MNIST 在线合成数据集.

按论文原始设定 (Srivastava et al., 2015) 从静态 MNIST 生成运动视频序列:
  - 每条序列随机取 n_digits 个数字, 放入 canvas_size x canvas_size 黑色画布
  - 每个数字随机初始位置 + 随机方向速度 (speed_range 像素/帧)
  - 撞到画布边缘反弹, 数字之间重叠时取像素最大值
  - 生成 seq_len 帧

每次调用都随机生成, 相当于无限量的训练数据; 传入 seed 可得到可复现的序列
(适合做固定的验证/测试集).

用法:
  from moving_mnist import MovingMNIST
  ds = MovingMNIST(root=r"D:\MovingMNIST\dataset\data")
  sample = ds[0]
  sample["video"]   : FloatTensor (T, 1, H, W), 像素 [0, 1]
  sample["labels"]  : LongTensor (n_digits,)     每个数字的类别 0-9
  sample["centers"] : FloatTensor (T, n_digits, 2) 每帧每个数字的质心 (x, y) 像素坐标
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 该环境存在双 libiomp5md.dll, 须在 import torch 前设置

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets

DIGIT_SIZE = 28  # MNIST 原始数字尺寸


class MovingMNIST(Dataset):
    """在线合成 Moving MNIST 序列.

    Args:
        root:        torchvision MNIST 数据目录 (含 MNIST/raw/...)
        train:       用 MNIST 训练集还是测试集的数字
        n_digits:    每条序列的数字个数 (论文默认 2)
        canvas_size: 画布边长 (论文默认 64)
        seq_len:     帧数 (论文默认 20)
        speed_range: 速度范围 (像素/帧), 论文默认 (3, 5)
        length:      数据集"长度"——在线生成所以只是逻辑大小
        seed:        None=完全随机(训练用); 整数=按下标确定性生成(验证/测试用)
    """

    def __init__(self, root, train=True, n_digits=2, canvas_size=64, seq_len=20,
                 speed_range=(3, 5), length=10000, seed=None):
        self.mnist = datasets.MNIST(root=root, train=train, download=False)
        self.n_digits = n_digits
        self.canvas = canvas_size
        self.seq_len = seq_len
        self.speed_range = speed_range
        self.length = length
        self.seed = seed

        if canvas_size <= DIGIT_SIZE:
            raise ValueError("canvas_size 必须大于 28")

    def __len__(self):
        return self.length

    def _sample_sequence(self, rng):
        """生成一条序列, 返回 video (T,H,W) uint8, labels, centers (T,N,2)。"""
        H = W = self.canvas
        limit = H - DIGIT_SIZE  # 数字左上角的可活动范围 [0, limit]

        # --- 随机取数字 ---
        idxs = rng.randint(0, len(self.mnist), size=self.n_digits)
        digits, labels = [], []
        for i in idxs:
            img, lab = self.mnist[int(i)]
            digits.append(np.asarray(img, dtype=np.uint8))
            labels.append(lab)
        digits = np.stack(digits)                    # (N, 28, 28)

        # --- 随机初始位置与速度 ---
        pos = rng.randint(0, limit + 1, size=(self.n_digits, 2)).astype(np.float64)
        angles = rng.uniform(0, 2 * np.pi, size=self.n_digits)
        speeds = rng.uniform(*self.speed_range, size=self.n_digits)
        vel = np.stack([np.cos(angles), np.sin(angles)], axis=1) * speeds[:, None]

        # --- 逐帧推演 ---
        frames = np.zeros((self.seq_len, H, W), dtype=np.uint8)
        centers = np.zeros((self.seq_len, self.n_digits, 2), dtype=np.float32)
        half = DIGIT_SIZE / 2.0
        for t in range(self.seq_len):
            for n in range(self.n_digits):
                x, y = int(pos[n, 0]), int(pos[n, 1])
                # 重叠区域取最大值 (论文做法)
                np.maximum(frames[t, y:y + DIGIT_SIZE, x:x + DIGIT_SIZE],
                           digits[n], out=frames[t, y:y + DIGIT_SIZE, x:x + DIGIT_SIZE])
                centers[t, n] = (pos[n, 0] + half, pos[n, 1] + half)

            pos += vel
            # 撞墙反弹: 位置折回界内, 对应方向速度取反
            for axis in range(2):
                over = pos[:, axis] > limit
                under = pos[:, axis] < 0
                pos[over, axis] = 2 * limit - pos[over, axis]
                pos[under, axis] = -pos[under, axis]
                vel[over | under, axis] *= -1

        return frames, np.asarray(labels, dtype=np.int64), centers

    def __getitem__(self, idx):
        # 训练模式: 每次随机; 固定 seed 模式: 同一下标永远生成同一序列
        rng = np.random.RandomState(self.seed + idx) if self.seed is not None else np.random
        video, labels, centers = self._sample_sequence(rng)

        video = torch.from_numpy(video).float().div_(255.0).unsqueeze(1)  # (T,1,H,W)
        return {
            "video": video,
            "labels": torch.from_numpy(labels),
            "centers": torch.from_numpy(centers),  # (T,N,2) 像素坐标, 原点在左上
        }


def save_gif(video, path, labels=None, fps=5):
    """把 (T,1,H,W) 张量存成 gif, 便于肉眼检查生成效果。"""
    from PIL import Image
    frames = [(video[t, 0].numpy() * 255).astype(np.uint8) for t in range(video.shape[0])]
    imgs = [Image.fromarray(f).resize((f.shape[1] * 4, f.shape[0] * 4), Image.NEAREST) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=1000 // fps, loop=0)


if __name__ == "__main__":
    ROOT = r"D:\MovingMNIST\dataset\data"
    out_dir = r"D:\MovingMNIST\samples"
    os.makedirs(out_dir, exist_ok=True)

    ds = MovingMNIST(root=ROOT, seed=42)  # 固定 seed, 演示可复现
    print("dataset length:", len(ds))

    for i in range(3):
        s = ds[i]
        print(f"[{i}] video {tuple(s['video'].shape)}  labels {s['labels'].tolist()}"
              f"  首帧质心 {np.round(s['centers'][0].numpy(), 1).tolist()}"
              f"  末帧质心 {np.round(s['centers'][-1].numpy(), 1).tolist()}")
        save_gif(s["video"], os.path.join(out_dir, f"sample_{i}.gif"))

    print("gif 已保存到", out_dir)
