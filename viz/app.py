"""
Stewart 平台实时监控台。

- 左：传感器强度热力图 / 球位置分布密度热力图
- 右：当前帧关键指标与告警
- 下：球位置与平台姿态的时间趋势
- 侧边栏：数据源、刷新节奏、时间窗口、位置筛选

启动：streamlit run app.py
"""
from __future__ import annotations

import math
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from data_source import (
    BALL_THRESHOLD,
    GRID,
    GRID_X,
    GRID_Y,
    HALF_SPAN_MM,
    SPACING_MM,
    DataReader,
)

st.set_page_config(
    page_title="Stewart 平台实时监控",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="expanded",
)

FONT = "Microsoft YaHei, Segoe UI, sans-serif"
ACCENT = "#00E5FF"
WARN = "#FFB020"
DANGER = "#FF5252"
OK = "#3DDC97"


# ---------------------------------------------------------------------------
# 侧边栏
# ---------------------------------------------------------------------------

def sidebar() -> dict:
    with st.sidebar:
        st.markdown("### ◎ Stewart 监控台")
        st.caption("三支链平衡平台 · 实时数据可视化")

        st.divider()
        st.markdown("**数据源**")
        mode_label = st.radio(
            "模式", ["模拟数据", "串口设备", "CSV 回放"],
            index=0, label_visibility="collapsed",
            help="无硬件时用模拟数据；串口需 Arduino 持续输出；CSV 用于复盘",
        )

        port = None
        baud = 115200
        csv_path = None

        if mode_label == "串口设备":
            ports = []
            try:
                from serial.tools import list_ports
                ports = [p.device for p in list_ports.comports()]
            except Exception:
                pass
            if ports:
                port = st.selectbox("端口", ports, index=0)
            else:
                port = st.text_input("端口", value="COM3", help="未检测到串口设备，可手动填写")
            baud = st.selectbox("波特率", [115200, 9600, 57600, 230400], index=0)

        elif mode_label == "CSV 回放":
            up = st.file_uploader("上传 CSV", type=["csv"], help="需包含 t, x, y 列，可选 roll/pitch/ir0..ir15")
            if up is not None:
                tmp = Path(tempfile.gettempdir()) / "stewart_replay.csv"
                tmp.write_bytes(up.getvalue())
                csv_path = str(tmp)
                st.caption(f"已载入：{up.name}")

        st.divider()
        st.markdown("**刷新**")
        auto = st.toggle("自动刷新", value=True)
        refresh = st.select_slider("间隔", options=[0.3, 0.5, 1.0, 2.0], value=0.5,
                                   format_func=lambda v: f"{v:g}s", disabled=not auto)

        st.divider()
        st.markdown("**时间范围**")
        window = st.select_slider("窗口", options=[10, 20, 30, 60, 120, 300], value=60,
                                  format_func=lambda v: f"最近 {v}s")

        st.markdown("**位置筛选**")
        region = st.selectbox("区域", ["全部区域", "中心区", "边缘区", "自定义半径"])
        radius = 15.0
        if region == "中心区":
            radius = 15.0
        elif region == "边缘区":
            radius = 15.0
        elif region == "自定义半径":
            radius = st.slider("半径 (mm)", 5.0, float(HALF_SPAN_MM), 20.0, 1.0)

        st.divider()
        st.markdown("**曲线指标**")
        show_pos = st.checkbox("球位置 X / Y", value=True)
        show_att = st.checkbox("平台姿态 roll / pitch", value=True)

    mode = {"模拟数据": "sim", "串口设备": "serial", "CSV 回放": "csv"}[mode_label]
    return {
        "mode": mode, "port": port, "baud": baud, "csv_path": csv_path,
        "auto": auto, "refresh": refresh, "window": window,
        "region": region, "radius": radius,
        "show_pos": show_pos, "show_att": show_att,
    }


def get_reader(cfg: dict) -> DataReader:
    """按配置复用采集器；参数变化时先停旧实例，避免串口被占用。"""
    key = (cfg["mode"], cfg["port"], cfg["baud"], cfg["csv_path"])
    if st.session_state.get("_reader_key") != key:
        old = st.session_state.get("_reader")
        if old is not None:
            old.stop()
        st.session_state["_reader"] = DataReader(
            mode=cfg["mode"], port=cfg["port"], baud=cfg["baud"], csv_path=cfg["csv_path"]
        )
        st.session_state["_reader_key"] = key
    return st.session_state["_reader"]


# ---------------------------------------------------------------------------
# 图表
# ---------------------------------------------------------------------------

def heatmap_sensor(frame, radius_filter: float | None) -> go.Figure:
    """当前帧 16 路强度热力图，叠加球位、目标点与筛选区域。"""
    z = np.asarray(frame.ir, dtype=float).reshape(GRID, GRID)
    zmax = max(600.0, float(z.max()))
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=z, x=GRID_X, y=GRID_Y,
        colorscale="Inferno", zmin=0, zmax=zmax,
        colorbar=dict(title="强度", thickness=10, len=0.75, outlinewidth=0),
        hovertemplate="X %{x:.0f} mm<br>Y %{y:.0f} mm<br>强度 %{z:.0f}<extra></extra>",
        name="传感器",
    ))
    if radius_filter:
        fig.add_shape(
            type="circle", x0=-radius_filter, y0=-radius_filter,
            x1=radius_filter, y1=radius_filter,
            line=dict(color=OK, width=1.4, dash="dot"),
            fillcolor="rgba(61,220,151,0.06)",
        )
    fig.add_trace(go.Scatter(
        x=[frame.x], y=[frame.y], mode="markers", name="球位置",
        marker=dict(size=16, color=ACCENT, line=dict(color="white", width=2)),
        hovertemplate="球 X %{x:.1f} mm<br>球 Y %{y:.1f} mm<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[frame.sx], y=[frame.sy], mode="markers", name="目标",
        marker=dict(size=15, symbol="x-thin", color=DANGER, line=dict(width=3)),
        hovertemplate="目标 X %{x:.1f} mm<br>目标 Y %{y:.1f} mm<extra></extra>",
    ))
    fig.update_layout(
        height=420, template="plotly_dark",
        margin=dict(l=8, r=8, t=34, b=8),
        title=dict(text="传感器强度分布", font=dict(size=14, family=FONT)),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(size=11)),
        font=dict(family=FONT, size=11),
    )
    fig.update_xaxes(range=[-52, 52], dtick=30, title="X (mm)",
                     gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.15)")
    fig.update_yaxes(range=[-52, 52], dtick=30, title="Y (mm)",
                     gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.15)",
                     scaleanchor="x", scaleratio=1)
    return fig


def heatmap_density(df: pd.DataFrame, radius_filter: float | None) -> go.Figure:
    """窗口内球位置二维密度，观察停留区域。"""
    fig = go.Figure()
    if len(df) >= 2:
        fig.add_trace(go.Histogram2d(
            x=df["x"], y=df["y"],
            xbins=dict(start=-45, end=45, size=5),
            ybins=dict(start=-45, end=45, size=5),
            colorscale="Viridis",
            colorbar=dict(title="帧数", thickness=10, len=0.75, outlinewidth=0),
            hovertemplate="X %{x:.0f} mm<br>Y %{y:.0f} mm<br>帧数 %{z}<extra></extra>",
        ))
    if radius_filter:
        fig.add_shape(
            type="circle", x0=-radius_filter, y0=-radius_filter,
            x1=radius_filter, y1=radius_filter,
            line=dict(color=OK, width=1.4, dash="dot"),
            fillcolor="rgba(61,220,151,0.06)",
        )
    fig.update_layout(
        height=420, template="plotly_dark",
        margin=dict(l=8, r=8, t=34, b=8),
        title=dict(text=f"球位置分布密度（{len(df)} 帧）", font=dict(size=14, family=FONT)),
        font=dict(family=FONT, size=11),
    )
    fig.update_xaxes(range=[-52, 52], dtick=30, title="X (mm)",
                     gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(range=[-52, 52], dtick=30, title="Y (mm)",
                     gridcolor="rgba(255,255,255,0.08)",
                     scaleanchor="x", scaleratio=1)
    return fig


def timeseries(df: pd.DataFrame, show_pos: bool, show_att: bool) -> go.Figure:
    cols = int(show_pos) + int(show_att)
    cols = max(1, cols)
    titles = []
    if show_pos:
        titles.append("球位置 (mm)")
    if show_att:
        titles.append("平台姿态 (°)")
    fig = make_subplots(rows=1, cols=cols, subplot_titles=titles, horizontal_spacing=0.09)

    idx = 1
    if show_pos:
        fig.add_trace(go.Scatter(x=df["time"], y=df["x"], name="球 X",
                                 line=dict(color=ACCENT, width=1.5)), row=1, col=idx)
        fig.add_trace(go.Scatter(x=df["time"], y=df["y"], name="球 Y",
                                 line=dict(color="#FF9F43", width=1.5)), row=1, col=idx)
        fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.25)", width=1), row=1, col=idx)
        idx += 1
    if show_att:
        fig.add_trace(go.Scatter(x=df["time"], y=df["roll"], name="roll",
                                 line=dict(color="#7CFFB2", width=1.5)), row=1, col=idx)
        fig.add_trace(go.Scatter(x=df["time"], y=df["pitch"], name="pitch",
                                 line=dict(color="#C792EA", width=1.5)), row=1, col=idx)
        fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.25)", width=1), row=1, col=idx)

    fig.update_layout(
        height=300, template="plotly_dark",
        margin=dict(l=8, r=8, t=40, b=8),
        legend=dict(orientation="h", yanchor="bottom", y=1.12, x=0, font=dict(size=11)),
        font=dict(family=FONT, size=11),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.07)", tickformat="%H:%M:%S")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.07)")
    return fig


# ---------------------------------------------------------------------------
# 诊断
# ---------------------------------------------------------------------------

def diagnose(frames: list, df: pd.DataFrame, stats: dict, now: float) -> list[tuple[str, str]]:
    issues: list[tuple[str, str]] = []

    if stats.get("error"):
        issues.append(("error", stats["error"]))
    if not frames:
        if stats.get("uptime", 99) < 4:
            issues.append(("info", "正在连接数据源，等待首帧…"))
        else:
            issues.append(("error", "尚未收到任何数据。请检查数据源配置，或切换到「模拟数据」验证界面。"))
        return issues

    last = frames[-1]
    gap = now - last.t
    if gap > 2.0:
        issues.append(("warning", f"数据已中断 {gap:.1f}s（最后一条 {datetime.fromtimestamp(last.t):%H:%M:%S}）。"
                                  "串口模式下请确认设备在线、波特率一致。"))

    ir = np.asarray(last.ir, dtype=float)
    if not np.all(np.isfinite(ir)):
        issues.append(("warning", "传感器数据含 NaN/Inf，已从统计中剔除。"))
    if ir.max() - ir.min() < BALL_THRESHOLD:
        issues.append(("info", f"当前帧未检测到球（max−min = {ir.max() - ir.min():.0f} < 阈值 {BALL_THRESHOLD:.0f}）。"))

    r = math.hypot(last.x, last.y)
    if r > HALF_SPAN_MM:
        issues.append(("warning", f"球位置 r = {r:.1f} mm 已超出检测区 ±{HALF_SPAN_MM:.0f} mm，坐标不可信。"))
    if not last.valid:
        issues.append(("info", last.note or "当前帧标记为无效。"))

    if len(df) >= 5 and df["x"].notna().sum() == 0:
        issues.append(("warning", "窗口内没有有效位置样本，曲线为空。"))

    dropped = stats.get("dropped", 0)
    total = stats.get("total", 0)
    if total + dropped > 50 and dropped / (total + dropped) > 0.2:
        issues.append(("warning", f"解析失败率 {dropped / (total + dropped):.0%}，请检查数据格式是否与固件输出一致。"))

    return issues


# ---------------------------------------------------------------------------
# 主渲染
# ---------------------------------------------------------------------------

def render(cfg: dict) -> None:
    reader = get_reader(cfg)
    frames = reader.snapshot()
    stats = reader.stats()
    now = time.time()

    raw = pd.DataFrame([f.as_row() for f in frames]) if frames else pd.DataFrame()

    df = raw.copy()
    if not df.empty:
        df = df[df["t"] >= now - cfg["window"]].copy()
        df["time"] = [datetime.fromtimestamp(v) for v in df["t"]]

        # 位置筛选
        r = np.hypot(df["x"], df["y"])
        if cfg["region"] == "中心区":
            df = df[r <= cfg["radius"]]
        elif cfg["region"] == "边缘区":
            df = df[r > cfg["radius"]]
        elif cfg["region"] == "自定义半径":
            df = df[r <= cfg["radius"]]

    # ---- 顶部状态 ----
    top = st.columns([2.2, 1, 1, 1, 1])
    with top[0]:
        mode_txt = {"sim": "模拟数据", "serial": "串口设备", "csv": "CSV 回放"}[cfg["mode"]]
        if stats["error"]:
            st.markdown(f"#### ◎ Stewart 监控台　<span style='color:{DANGER};font-size:14px'>● 异常</span>", unsafe_allow_html=True)
        elif frames and now - frames[-1].t <= 2:
            st.markdown(f"#### ◎ Stewart 监控台　<span style='color:{OK};font-size:14px'>● 在线</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"#### ◎ Stewart 监控台　<span style='color:{WARN};font-size:14px'>● 等待数据</span>", unsafe_allow_html=True)
        st.caption(f"{mode_txt} · 缓冲 {stats['buffered']} 帧 · 窗口 {cfg['window']}s")

    total = stats["total"] + stats["dropped"]
    rate = stats["total"] / max(1e-6, stats["uptime"])
    quality = 1 - (stats["dropped"] / total) if total else 1.0
    with top[1]:
        st.metric("采样率", f"{rate:.1f} Hz")
    with top[2]:
        st.metric("数据质量", f"{quality:.1%}")
    with top[3]:
        st.metric("窗口帧数", f"{len(df):,}")
    with top[4]:
        if frames:
            st.metric("最新延迟", f"{max(0.0, now - frames[-1].t):.2f} s")
        else:
            st.metric("最新延迟", "—")

    st.divider()

    # ---- 主体：热力图 + 当前值 ----
    left, right = st.columns([2.35, 1], gap="medium")

    with left:
        if frames:
            tab1, tab2 = st.tabs(["传感器强度", "位置分布密度"])
            rf = cfg["radius"] if cfg["region"] in ("中心区", "边缘区", "自定义半径") else None
            with tab1:
                st.plotly_chart(heatmap_sensor(frames[-1], rf), width="stretch",
                                config={"displayModeBar": False})
            with tab2:
                st.plotly_chart(heatmap_density(df, rf), width="stretch",
                                config={"displayModeBar": False})
        else:
            st.info("等待首帧数据…")

    with right:
        st.markdown("**当前帧**")
        if frames:
            f = frames[-1]
            r = math.hypot(f.x, f.y)
            c1, c2 = st.columns(2)
            c1.metric("球 X", f"{f.x:+.1f} mm")
            c2.metric("球 Y", f"{f.y:+.1f} mm")
            c3, c4 = st.columns(2)
            c3.metric("距中心", f"{r:.1f} mm")
            c4.metric("roll", f"{f.roll:+.2f}°")
            c5, c6 = st.columns(2)
            c5.metric("pitch", f"{f.pitch:+.2f}°")
            c6.metric("强度峰值", f"{max(f.ir):.0f}")

            st.markdown("**状态**")
            if r <= 10:
                st.success("球位居中，控制稳定")
            elif r <= HALF_SPAN_MM:
                st.info("球在检测区内，正在回中")
            else:
                st.warning("球已离开检测区")

            # 16 路强度条
            st.markdown("**16 路强度**")
            bar = go.Figure(go.Bar(
                x=list(range(16)), y=f.ir,
                marker=dict(color=f.ir, colorscale="Inferno", cmin=0, cmax=max(600, max(f.ir))),
            ))
            bar.update_layout(height=130, template="plotly_dark", showlegend=False,
                              margin=dict(l=0, r=0, t=4, b=0),
                              font=dict(family=FONT, size=9))
            bar.update_xaxes(showticklabels=False, showgrid=False)
            bar.update_yaxes(showgrid=False)
            st.plotly_chart(bar, width="stretch", config={"displayModeBar": False})
        else:
            st.caption("—")

    # ---- 时间序列 ----
    st.divider()
    if frames and len(df) > 1 and (cfg["show_pos"] or cfg["show_att"]):
        st.plotly_chart(timeseries(df, cfg["show_pos"], cfg["show_att"]),
                        width="stretch", config={"displayModeBar": False})
    elif frames:
        st.info("当前筛选条件下样本不足，曲线暂不显示。可放宽时间窗口或位置筛选。")

    # ---- 告警 ----
    issues = diagnose(frames, df, stats, now)
    if issues:
        st.markdown("**诊断提示**")
        for level, msg in issues:
            {"error": st.error, "warning": st.warning, "info": st.info}[level](msg)
    else:
        st.success("数据链路正常，未检测到异常。")

    # ---- 底部操作 ----
    b1, b2, _ = st.columns([1, 1, 5])
    with b1:
        if st.button("立即刷新", width="stretch"):
            st.rerun(scope="fragment")
    with b2:
        if st.button("清空缓冲", width="stretch"):
            reader.frames.clear()
            st.rerun(scope="fragment")


cfg = sidebar()

# 局部刷新：只重绘数据区，侧边栏与页面骨架保持稳定
@st.fragment(run_every=(cfg["refresh"] if cfg["auto"] else None))
def live_panel() -> None:
    render(cfg)


live_panel()
