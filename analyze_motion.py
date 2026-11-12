"""分析观测段(前10帧)数字的运动模式, 为方案B2(分段外推+反弹检测)提供依据.

关键问题:
  1. 观测段内每个数字是否真的匀速? (速度方差 vs 均值)
  2. 观测段内发生反弹的频率? (反弹会破坏"末帧速度=未来速度"假设)
  3. 未来段(后10帧)反弹的频率? 末帧朝向边界时, ref=q+v 是否会被反弹打脸?
  4. 末帧两帧差分速度 vs 整段拟合速度, 哪个更接近未来真实速度?

产出: results/运动模式分析.md + 终端报告
运行: E:\\anaconda3\\envs\\CurveSegmentationMethod\\python.exe analyze_motion.py
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moving_mnist import MovingMNIST

DATA_ROOT = r"D:\MovingMNIST\dataset\data"
OUT_DIR = r"D:\MovingMNIST\results"
N_INPUT, N_FUTURE = 10, 10
N_EVAL = 2000
TEST_SEED = 2026
LIMIT = 64 - 28  # 活动范围 [0, 36], 数字左上角坐标


def detect_bounces(centers):
    """检测每个数字在 [0, T-1] 帧内是否发生反弹(速度方向突变).

    用相邻帧差分近似速度, 若相邻两帧速度在某轴上符号翻转, 视为一次反弹.
    centers (B,T,N,2). 返回 bounce (B,T-1,N) bool: 该帧到下一帧之间发生反弹.
    """
    B, T, N, _ = centers.shape
    vel = np.diff(centers, axis=1)  # (B,T-1,N,2) 每帧速度
    # 相邻速度点积为负 -> 反弹 (某轴反转)
    dot = (vel[:, :-1] * vel[:, 1:]).sum(axis=-1)  # (B,T-2,N)
    bounce = np.zeros((B, T - 1, N), dtype=bool)
    bounce[:, :-1] = dot < 0
    return bounce, vel


def main():
    ds = MovingMNIST(root=DATA_ROOT, train=False, seq_len=N_INPUT + N_FUTURE,
                     length=N_EVAL, seed=TEST_SEED)
    # 收集所有序列的质心
    all_centers = []
    for i in range(N_EVAL):
        all_centers.append(ds[i]["centers"].numpy())
    centers = np.stack(all_centers)  # (B, T, N, 2)
    B, T, N, _ = centers.shape

    obs = centers[:, :N_INPUT]      # (B,10,N,2)
    fut = centers[:, N_INPUT:]      # (B,10,N,2)

    # --- 1. 观测段速度一致性 ---
    obs_vel = np.diff(obs, axis=1)  # (B,9,N,2)
    obs_speed = np.linalg.norm(obs_vel, axis=-1)  # (B,9,N)
    speed_mean = obs_speed.mean(axis=1)           # (B,N)
    speed_std = obs_speed.std(axis=1)
    # 相对变异系数
    rel_std = speed_std / (speed_mean + 1e-6)

    # --- 2. 反弹统计 ---
    bounce, vel_full = detect_bounces(centers)    # (B,T-1,N)
    bounce_obs = bounce[:, :N_INPUT - 1].sum(axis=1)       # 观测段每序列反弹次数 (B,N)
    bounce_fut = bounce[:, N_INPUT - 1:].sum(axis=1)       # 未来段每序列反弹次数 (B,N)
    n_obs_bounce = (bounce_obs.sum(axis=1) > 0).sum()      # 观测段发生过反弹的序列数
    n_fut_bounce = (bounce_fut.sum(axis=1) > 0).sum()      # 未来段发生过反弹的序列数

    # --- 3. 末帧差分速度 vs 拟合速度 ---
    # 末帧两帧差分速度 (planB 当前用)
    v_last = obs[:, -1] - obs[:, -2]            # (B,N,2)
    # 观测段整段拟合速度 (匀速假设, 线性拟合)
    t = np.arange(N_INPUT, dtype=np.float64)
    A = np.stack([t, np.ones_like(t)], axis=1)  # (T,2)
    # 对每个轴做最小二乘: pos = v*t + b
    fitted_v = np.zeros((B, N, 2))
    for b in range(B):
        for n in range(N):
            for ax in range(2):
                coef, *_ = np.linalg.lstsq(A, obs[b, :, n, ax], rcond=None)
                fitted_v[b, n, ax] = coef[0]

    # 未来第1帧真实速度 (真值, 用于评估哪个外推更准)
    v_fut_true = fut[:, 0] - obs[:, -1]          # (B,N,2) 观测末帧到未来首帧位移
    err_last = np.linalg.norm(v_last - v_fut_true, axis=-1).mean()
    err_fit = np.linalg.norm(fitted_v - v_fut_true, axis=-1).mean()

    # --- 4. 朝向边界的末帧: 外推越过边界则未来会反弹 ---
    # 从观测末帧位置 + 末帧速度外推一步
    q_last = obs[:, -1]                          # (B,N,2)
    ref_last = q_last + v_last                   # (B,N,2) 外推位置
    # 是否会越过边界 (考虑左上角范围 [0,36])
    over_bound = (ref_last > LIMIT) | (ref_last < 0)
    n_near_bound = over_bound.any(axis=-1).sum()  # 外推越过边界的 (序列,数字) 数

    lines = [
        "# 观测段运动模式分析\n\n",
        f"- 测试集: seed={TEST_SEED} x {B} 序列 x {N} 数字, 观测 {N_INPUT} 帧 / 未来 {N_FUTURE} 帧\n",
        f"- 活动范围: 左上角 [0, {LIMIT}], 速度范围 (3,5) px/帧, 撞墙反弹\n\n",
        "## 1. 速度一致性 (观测段)\n\n",
        f"- 观测段速度相对变异系数 (std/mean) 均值: {rel_std.mean():.3f}\n",
        f"- 说明: 接近 0 表示匀速, 越大表示速度波动/反弹越多\n\n",
        "## 2. 反弹统计\n\n",
        f"- 观测段发生过反弹的序列: {n_obs_bounce}/{B} ({n_obs_bounce/B*100:.1f}%)\n",
        f"- 未来段发生过反弹的序列: {n_fut_bounce}/{B} ({n_fut_bounce/B*100:.1f}%)\n",
        f"- 观测段每序列平均反弹次数: {bounce_obs.sum()/B:.3f}\n",
        f"- 未来段每序列平均反弹次数: {bounce_fut.sum()/B:.3f}\n\n",
        "## 3. 速度外推哪个更准 (对比未来首帧真实位移)\n\n",
        f"- 末帧两帧差分速度 平均误差: {err_last:.3f} px\n",
        f"- 观测段整段拟合速度 平均误差: {err_fit:.3f} px\n",
        f"- 结论: {'拟合更优' if err_fit < err_last else '差分更优'}\n\n",
        "## 4. 末帧朝向边界的比例\n\n",
        f"- 外推一步越过边界的 (序列,数字) 数: {n_near_bound}/{B*N} ({n_near_bound/(B*N)*100:.1f}%)\n",
        f"- 这些场景 ref=q+v 会被反弹打脸, 需反弹检测纠正\n",
    ]

    with open(os.path.join(OUT_DIR, "运动模式分析.md"), "w", encoding="utf-8") as f:
        f.writelines(lines)

    print("\n===== 观测段运动模式分析 =====")
    print(f"序列数 {B}, 每序列 {N} 数字, 观测 {N_INPUT} 帧")
    print(f"[速度一致性] 相对变异系数均值 {rel_std.mean():.3f} (0=完全匀速)")
    print(f"[反弹] 观测段 {n_obs_bounce}/{B} ({n_obs_bounce/B*100:.1f}%) "
          f"| 未来段 {n_fut_bounce}/{B} ({n_fut_bounce/B*100:.1f}%)")
    print(f"[外推] 末帧差分误差 {err_last:.3f} px vs 整段拟合误差 {err_fit:.3f} px "
          f"-> {'拟合更优' if err_fit < err_last else '差分更优'}")
    print(f"[边界] 外推越过边界的数字 {n_near_bound}/{B*N} ({n_near_bound/(B*N)*100:.1f}%)")
    print(f"\n报告 -> {os.path.join(OUT_DIR, '运动模式分析.md')}")


if __name__ == "__main__":
    main()
