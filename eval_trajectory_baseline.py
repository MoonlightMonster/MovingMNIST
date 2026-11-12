"""步骤 2: 轨迹预测基线 —— 线性匀速外推 (无需训练).

任务设定: 给定前 n_input 帧的质心, 预测后 n_future 帧的质心.
方法: 用观测窗口内的平均速度做匀速直线外推
      v = (c_last - c_first) / (n_input - 1)
      c(t+k) = c_last + k * v
这是不含任何学习成分的数学基线, 是后续学习模型必须超越的底线.
注意: Moving MNIST 数字撞墙会反弹, 纯线性外推在反弹后必然出错,
      这正是学习模型应该击败它的地方.

指标:
  ADE (Average Displacement Error)  : 预测段所有帧的平均欧氏距离 (像素)
  FDE (Final Displacement Error)    : 最后一帧的欧氏距离 (像素)

运行:
  E:\\anaconda3\\envs\\CurveSegmentationMethod\\python.exe eval_trajectory_baseline.py
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys

import numpy as np

sys.path.insert(0, r"D:\MovingMNIST")
from moving_mnist import MovingMNIST

DATA_ROOT = r"D:\MovingMNIST\dataset\data"
OUT_DIR = r"D:\MovingMNIST\samples"

N_INPUT = 10      # 观测帧数
N_FUTURE = 10     # 预测帧数
N_EVAL = 2000     # 评估序列数 (固定 seed, 结果可复现)
EVAL_SEED = 2026


def linear_extrapolate(obs):
    """obs: (n_input, 2) 质心序列 -> (n_future, 2) 匀速外推."""
    v = (obs[-1] - obs[0]) / (len(obs) - 1)
    k = np.arange(1, N_FUTURE + 1, dtype=np.float64)[:, None]
    return obs[-1] + k * v


def main():
    ds = MovingMNIST(root=DATA_ROOT, seq_len=N_INPUT + N_FUTURE,
                     length=N_EVAL, seed=EVAL_SEED)

    ade_all, fde_all = [], []
    for i in range(len(ds)):
        centers = ds[i]["centers"].numpy().astype(np.float64)  # (T, N, 2)
        for n in range(centers.shape[1]):                      # 每个数字独立预测
            obs, future = centers[:N_INPUT, n], centers[N_INPUT:, n]
            pred = linear_extrapolate(obs)
            err = np.linalg.norm(pred - future, axis=1)        # (n_future,)
            ade_all.append(err.mean())
            fde_all.append(err[-1])
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(ds)} ...")

    ade, fde = float(np.mean(ade_all)), float(np.mean(fde_all))
    canvas = ds.canvas
    print(f"\n=== 线性外推基线 ({len(ds)} 条序列 x {ds.n_digits} 个数字) ===")
    print(f"ADE: {ade:.2f} px  ({ade / canvas * 100:.1f}% 画布边长)")
    print(f"FDE: {fde:.2f} px  ({fde / canvas * 100:.1f}% 画布边长)")

    # ---- 可视化一个样本: 观测轨迹(白) / 真实未来(绿) / 线性外推(红) ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = ds[0]
    video = s["video"].numpy()
    centers = s["centers"].numpy()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, t in zip(axes, [0, 9, 19]):
        ax.imshow(video[t, 0], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"frame {t}")
        ax.axis("off")
        for n, color in zip(range(centers.shape[1]), ["cyan", "yellow"]):
            obs, future = centers[:N_INPUT, n], centers[N_INPUT:, n]
            pred = linear_extrapolate(obs)
            ax.plot(obs[:, 0], obs[:, 1], "-o", color="white", ms=3, lw=1, label="observed" if n == 0 else None)
            ax.plot(future[:, 0], future[:, 1], "-o", color="lime", ms=3, lw=1, label="true future" if n == 0 else None)
            ax.plot(pred[:, 0], pred[:, 1], "--x", color="red", ms=3, lw=1, label="linear pred" if n == 0 else None)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Linear extrapolation baseline: observed (white) / true (green) / predicted (red)")
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, "traj_baseline.png"), dpi=110, bbox_inches="tight")
    print("可视化已保存到", os.path.join(OUT_DIR, "traj_baseline.png"))


if __name__ == "__main__":
    main()
