# -*- coding: utf-8 -*-
"""
STK 星地链路访问数据解析与可视化流水线
-------------------------------------------------
功能：
  1. 解析 STK 导出的 Access 链路时隙表与 AER 几何时序数据
  2. UTC 时间校准、数据清洗与排序
  3. 生成标准化链路过境时隙表
  4. 基于自由空间路径损耗 (FSPL) 模型离线计算信道增益与电波传播时延
  5. 链路指标统计与时序曲线可视化

输入文件：
  - access链路时隙表.csv
  - aer几何时序数据.csv
输出文件：
  - 标准化链路过境时隙表.csv
  - 链路参数时序数据.csv
  - 链路时序曲线.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ---------- 全局参数 ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ACCESS_CSV = os.path.join(BASE_DIR, "access链路时隙表.csv")
AER_CSV = os.path.join(BASE_DIR, "aer几何时序数据.csv")
OUT_ACCESS = os.path.join(BASE_DIR, "标准化链路过境时隙表.csv")
OUT_LINK = os.path.join(BASE_DIR, "链路参数时序数据.csv")
OUT_FIG = os.path.join(BASE_DIR, "链路时序曲线.png")

C = 2.99792458e8              # 真空光速 m/s
FREQ_HZ = 20e9                # 载波频率 20 GHz (Ka 波段，NTN 典型下行频率，可按需调整)
WAVELENGTH = C / FREQ_HZ      # 波长 m

# STK 时间格式: "15 Sep 2026 05:50:18.127"
STK_TIME_FMT = "%d %b %Y %H:%M:%S.%f"

# 中文字体配置
rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


# ---------- 1. 数据读取与 UTC 时间校准 ----------
def parse_stk_time(series: pd.Series) -> pd.Series:
    """将 STK 的 UTCG 字符串解析为带时区的 UTC datetime。"""
    return pd.to_datetime(series, format=STK_TIME_FMT, utc=True)


def load_access(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "Access": "access_id",
        "Start Time (UTCG)": "start_utc",
        "Stop Time (UTCG)": "stop_utc",
        "Duration (sec)": "duration_s",
    })
    df["start_utc"] = parse_stk_time(df["start_utc"])
    df["stop_utc"] = parse_stk_time(df["stop_utc"])
    df["duration_s"] = pd.to_numeric(df["duration_s"], errors="coerce")
    df = df.sort_values("start_utc").reset_index(drop=True)
    return df


def load_aer(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "Time (UTCG)": "time_utc",
        "Azimuth (deg)": "azimuth_deg",
        "Elevation (deg)": "elevation_deg",
        "Range (km)": "range_km",
    })
    df["time_utc"] = parse_stk_time(df["time_utc"])
    for col in ["azimuth_deg", "elevation_deg", "range_km"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # 数据清洗：去重 / 排序 / 剔除缺失
    df = df.dropna(subset=["time_utc", "range_km"]).drop_duplicates("time_utc")
    df = df.sort_values("time_utc").reset_index(drop=True)
    return df


# ---------- 2. 生成标准化链路过境时隙表 ----------
def build_access_table(access_df: pd.DataFrame, aer_df: pd.DataFrame) -> pd.DataFrame:
    """将 Access 与 AER 数据对齐，生成标准化过境时隙表。"""
    records = []
    for _, row in access_df.iterrows():
        mask = (aer_df["time_utc"] >= row["start_utc"]) & (aer_df["time_utc"] <= row["stop_utc"])
        sub = aer_df.loc[mask]
        records.append({
            "access_id": int(row["access_id"]),
            "start_utc": row["start_utc"],
            "stop_utc": row["stop_utc"],
            "duration_s": round(float(row["duration_s"]), 3),
            "sample_points": int(len(sub)),
            "az_min_deg": round(float(sub["azimuth_deg"].min()), 3) if len(sub) else None,
            "az_max_deg": round(float(sub["azimuth_deg"].max()), 3) if len(sub) else None,
            "el_min_deg": round(float(sub["elevation_deg"].min()), 3) if len(sub) else None,
            "el_max_deg": round(float(sub["elevation_deg"].max()), 3) if len(sub) else None,
            "range_min_km": round(float(sub["range_km"].min()), 3) if len(sub) else None,
            "range_max_km": round(float(sub["range_km"].max()), 3) if len(sub) else None,
        })
    return pd.DataFrame(records)


# ---------- 3. 自由空间损耗模型计算 ----------
def compute_link_params(aer_df: pd.DataFrame) -> pd.DataFrame:
    """基于 FSPL 模型计算信道增益与传播时延。"""
    df = aer_df.copy()
    d_m = df["range_km"].values * 1e3  # 距离换算为 m

    # 自由空间路径损耗 FSPL(dB) = 20*log10(4*pi*d/lambda)
    fspl_db = 20.0 * np.log10(4.0 * np.pi * d_m / WAVELENGTH)
    # 信道增益（路径增益，负值）= -FSPL
    channel_gain_db = -fspl_db
    # 电波传播时延 tau = d / c (ms)
    delay_ms = d_m / C * 1e3

    df["range_m"] = d_m
    df["fspl_db"] = np.round(fspl_db, 3)
    df["channel_gain_db"] = np.round(channel_gain_db, 3)
    df["prop_delay_ms"] = np.round(delay_ms, 3)

    # 为每条 AER 记录打上 access_id
    return df


def attach_access_id(aer_df: pd.DataFrame, access_df: pd.DataFrame) -> pd.DataFrame:
    """为每条 AER 时序记录关联其所属的 Access 编号。"""
    ids = np.full(len(aer_df), -1, dtype=int)
    for _, row in access_df.iterrows():
        mask = (aer_df["time_utc"] >= row["start_utc"]) & (aer_df["time_utc"] <= row["stop_utc"])
        ids[mask] = int(row["access_id"])
    aer_df["access_id"] = ids
    return aer_df


# ---------- 4. 链路指标统计 ----------
def link_statistics(aer_df: pd.DataFrame) -> pd.DataFrame:
    """按 Access 汇总链路关键指标统计。"""
    rows = []
    for aid, sub in aer_df[aer_df["access_id"] >= 0].groupby("access_id"):
        rows.append({
            "access_id": int(aid),
            "samples": int(len(sub)),
            "range_min_km": round(float(sub["range_km"].min()), 3),
            "range_max_km": round(float(sub["range_km"].max()), 3),
            "range_mean_km": round(float(sub["range_km"].mean()), 3),
            "fspl_min_db": round(float(sub["fspl_db"].min()), 3),
            "fspl_max_db": round(float(sub["fspl_db"].max()), 3),
            "fspl_mean_db": round(float(sub["fspl_db"].mean()), 3),
            "gain_max_db": round(float(sub["channel_gain_db"].max()), 3),
            "gain_min_db": round(float(sub["channel_gain_db"].min()), 3),
            "delay_min_ms": round(float(sub["prop_delay_ms"].min()), 3),
            "delay_max_ms": round(float(sub["prop_delay_ms"].max()), 3),
            "delay_mean_ms": round(float(sub["prop_delay_ms"].mean()), 3),
            "el_min_deg": round(float(sub["elevation_deg"].min()), 3),
            "el_max_deg": round(float(sub["elevation_deg"].max()), 3),
        })
    return pd.DataFrame(rows).sort_values("access_id").reset_index(drop=True)


# ---------- 5. 时序曲线可视化 ----------
def plot_link_timeseries(aer_df: pd.DataFrame, access_df: pd.DataFrame, out_path: str):
    """绘制仰角/斜距/FSPL/传播时延四联时序图，按 Access 分色。"""
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    cmap = plt.get_cmap("tab10")
    n_access = max(int(aer_df["access_id"].max()), 1) + 1

    for aid, sub in aer_df[aer_df["access_id"] >= 0].groupby("access_id"):
        color = cmap(int(aid) % 10)
        t = sub["time_utc"]
        axes[0].plot(t, sub["elevation_deg"], "-o", color=color, ms=4, lw=1.2,
                     label=f"Access {int(aid)}")
        axes[1].plot(t, sub["range_km"], "-o", color=color, ms=4, lw=1.2)
        axes[2].plot(t, sub["fspl_db"], "-o", color=color, ms=4, lw=1.2)
        axes[3].plot(t, sub["prop_delay_ms"], "-o", color=color, ms=4, lw=1.2)

    # 标注每次 Access 的起止时刻
    for _, row in access_df.iterrows():
        for ax in axes:
            ax.axvline(row["start_utc"], color="gray", ls="--", lw=0.6, alpha=0.5)
            ax.axvline(row["stop_utc"], color="gray", ls="--", lw=0.6, alpha=0.5)

    axes[0].set_ylabel("仰角 Elevation (deg)")
    axes[0].set_title("STK 星地链路过境几何参数时序曲线")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", ncol=3, fontsize=8)

    axes[1].set_ylabel("斜距 Range (km)")
    axes[1].grid(True, alpha=0.3)

    axes[2].set_ylabel(f"自由空间路径损耗 FSPL (dB)\n(f={FREQ_HZ/1e9:.2f} GHz)")
    axes[2].grid(True, alpha=0.3)

    axes[3].set_ylabel("传播时延 Delay (ms)")
    axes[3].set_xlabel("UTC 时间")
    axes[3].grid(True, alpha=0.3)

    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------- 主流程 ----------
def main():
    print("=" * 70)
    print("STK 星地链路访问数据解析与可视化流水线")
    print(f"载波频率: {FREQ_HZ/1e9:.2f} GHz | 波长: {WAVELENGTH*100:.2f} cm")
    print("=" * 70)

    # 1. 读取数据
    print("\n[1/5] 读取 STK 导出数据...")
    access_df = load_access(ACCESS_CSV)
    aer_df = load_aer(AER_CSV)
    print(f"  Access 过境次数: {len(access_df)}")
    print(f"  AER 采样点数  : {len(aer_df)}")
    print(f"  时间范围      : {aer_df['time_utc'].min()} ~ {aer_df['time_utc'].max()}")

    # 2. 生成标准化时隙表
    print("\n[2/5] 生成标准化链路过境时隙表...")
    std_access = build_access_table(access_df, aer_df)
    std_access.to_csv(OUT_ACCESS, index=False, encoding="utf-8-sig")
    print(f"  保存至: {OUT_ACCESS}")
    print(std_access.to_string(index=False))

    # 3. 关联 access_id 并计算链路参数
    print("\n[3/5] 基于自由空间损耗模型计算链路参数...")
    aer_df = attach_access_id(aer_df, access_df)
    aer_df = compute_link_params(aer_df)
    aer_df.to_csv(OUT_LINK, index=False, encoding="utf-8-sig")
    print(f"  保存至: {OUT_LINK}")
    print(aer_df[["time_utc", "access_id", "range_km", "fspl_db",
                  "channel_gain_db", "prop_delay_ms"]].head(8).to_string(index=False))

    # 4. 链路指标统计
    print("\n[4/5] 链路指标按过境统计...")
    stats = link_statistics(aer_df)
    print(stats.to_string(index=False))

    # 5. 可视化
    print("\n[5/5] 绘制时序曲线...")
    plot_link_timeseries(aer_df, access_df, OUT_FIG)
    print(f"  保存至: {OUT_FIG}")

    print("\n流水线执行完毕。")


if __name__ == "__main__":
    main()
