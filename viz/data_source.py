"""
Stewart 平台数据源。

三种模式：
  - sim    模拟源：带闭环物理的仿真，无硬件也能演示
  - serial 串口源：读取 Arduino 115200 输出
  - csv    CSV 回放：按时间戳还原历史数据

统一输出 Frame 对象，后台线程持续采集，供可视化层随时快照。
"""
from __future__ import annotations

import csv
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

# ---- 与固件 / README 保持一致的常量 ----
GRID = 4                    # 4x4 红外阵列
SPACING_MM = 30.0           # 传感器间距
HALF_SPAN_MM = 45.0         # 检测区半宽（4x4 @ 30mm）
BALL_THRESHOLD = 350.0      # 固件 BALL_THRESHOLD
ALPHA_SENSOR = 0.7          # 固件 IR_SMOOTH_ALPHA
ALPHA_OUTPUT = 0.9          # 固件 OUTPUT_SMOOTH_ALPHA
FPS = 30.0                  # 固件 30Hz 控制频率
G = 9810.0                  # 重力加速度 mm/s^2


@dataclass
class Frame:
    """一帧采样。坐标为平台坐标系，单位 mm，中心为原点。"""
    t: float
    ir: list[float]
    x: float
    y: float
    sx: float = 0.0
    sy: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    valid: bool = True
    note: str = ""

    def as_row(self) -> dict:
        row = {
            "t": self.t, "x": self.x, "y": self.y,
            "roll": self.roll, "pitch": self.pitch, "valid": self.valid,
        }
        for i, v in enumerate(self.ir):
            row[f"ir{i}"] = v
        return row


def sensor_positions() -> list[tuple[float, float]]:
    """16 个传感器的物理坐标，按行优先（y 从 +45 递减到 -45）。"""
    pos = []
    for row in range(GRID):
        for col in range(GRID):
            x = (col - (GRID - 1) / 2) * SPACING_MM
            y = ((GRID - 1) / 2 - row) * SPACING_MM
            pos.append((x, y))
    return pos


SENSOR_POS = sensor_positions()
GRID_X = [(c - (GRID - 1) / 2) * SPACING_MM for c in range(GRID)]
GRID_Y = [((GRID - 1) / 2 - r) * SPACING_MM for r in range(GRID)]


def synth_ir(x: float, y: float, amp: float = 900.0, sigma: float = 22.0,
             noise: float = 10.0, rng: Optional[np.random.Generator] = None) -> list[float]:
    """按球位置合成 16 路红外差分强度：高斯衰减 + 高斯噪声。"""
    rng = rng or np.random.default_rng()
    out = []
    for sx, sy in SENSOR_POS:
        d2 = (sx - x) ** 2 + (sy - y) ** 2
        v = amp * math.exp(-d2 / (2 * sigma ** 2)) + float(rng.normal(0, noise))
        out.append(max(0.0, v))
    return out


# ---------------------------------------------------------------------------
# 模拟源
# ---------------------------------------------------------------------------

class SimulatedSource:
    """
    闭环仿真：PID 由球位误差算平台倾角，倾角驱动球加速，形成"看得见在收敛"的运动。

    物理做了简化（忽略转动惯量与滚动摩擦细节），目的是让热力图与曲线呈现真实趋势，
    而非做控制器性能评估。
    """

    def __init__(self, seed: int = 7) -> None:
        self.rng = np.random.default_rng(seed)
        self.x = float(self.rng.uniform(-28, 28))
        self.y = float(self.rng.uniform(-28, 28))
        self.vx = self.vy = 0.0
        self.ix = self.iy = 0.0
        self.ex_prev = self.ey_prev = 0.0
        self.roll = self.pitch = 0.0
        self.ball_on = True
        self.off_since = 0.0
        self._last_ir: list[float] = [0.0] * 16
        self.last = time.time()
        self._disturb_at = time.time() + 7.0

    def step(self, dt_fixed: Optional[float] = None) -> Frame:
        now = time.time()
        dt = dt_fixed if dt_fixed else min(0.1, max(1e-3, now - self.last))
        self.last = now

        # 偶发扰动，模拟外部推球；小概率为强扰动，会把球推出检测区
        if now >= self._disturb_at:
            amp = 95.0 if self.rng.random() > 0.12 else 260.0
            self.vx += float(self.rng.normal(0, amp))
            self.vy += float(self.rng.normal(0, amp))
            self._disturb_at = now + float(self.rng.uniform(8, 14))

        # 持续微扰：平台振动、球面不平整带来的随机激励
        self.vx += float(self.rng.normal(0, 6.5))
        self.vy += float(self.rng.normal(0, 6.5))

        # 球运动：a = g * tan(theta)，带速度阻尼
        ax = G * math.tan(math.radians(self.roll))
        ay = G * math.tan(math.radians(self.pitch))
        damp = math.exp(-1.15 * dt)
        self.vx = (self.vx + ax * dt) * damp
        self.vy = (self.vy + ay * dt) * damp
        self.x += self.vx * dt
        self.y += self.vy * dt

        # 出界：球离开检测区，1.2s 后由人工放回
        if math.hypot(self.x, self.y) > HALF_SPAN_MM + 6:
            if self.ball_on:
                self.ball_on = False
                self.off_since = now
            elif now - self.off_since > 1.2:
                self.x = float(self.rng.uniform(-18, 18))
                self.y = float(self.rng.uniform(-18, 18))
                self.vx = self.vy = 0.0
                self.ball_on = True

        # PID：结构与固件一致，增益按 mm 量纲缩放
        ex, ey = -self.x, -self.y
        self.ix = max(-9.0, min(9.0, self.ix + 0.0022 * ex * dt * FPS))
        self.iy = max(-9.0, min(9.0, self.iy + 0.0022 * ey * dt * FPS))
        dx = (ex - self.ex_prev) * FPS
        dy = (ey - self.ey_prev) * FPS
        self.ex_prev, self.ey_prev = ex, ey

        kp, kd = 0.105, 0.011
        roll_raw = max(-15.0, min(15.0, kp * ex + self.ix + kd * dx))
        pitch_raw = max(-15.0, min(15.0, kp * ey + self.iy + kd * dy))
        self.roll = ALPHA_OUTPUT * roll_raw + (1 - ALPHA_OUTPUT) * self.roll
        self.pitch = ALPHA_OUTPUT * pitch_raw + (1 - ALPHA_OUTPUT) * self.pitch

        if self.ball_on:
            raw = synth_ir(self.x, self.y, rng=self.rng)
            ir = [ALPHA_SENSOR * r + (1 - ALPHA_SENSOR) * p for r, p in zip(raw, self._last_ir)]
            self._last_ir = ir
        else:
            ir = [0.0] * 16
            self._last_ir = ir

        return Frame(t=now, ir=ir, x=self.x, y=self.y, roll=self.roll,
                     pitch=self.pitch, valid=self.ball_on,
                     note="" if self.ball_on else "球已离开检测区")


# ---------------------------------------------------------------------------
# 串口源
# ---------------------------------------------------------------------------

class SerialSource:
    """
    解析 Arduino 输出，兼容固件里的两种格式：

      1) sendSerialData()  20 个逗号分隔值：ir0..ir15, cx, cy, sx, sy
         cx/cy 为格点单位（-1.5..2.5），乘以 30mm 转为平台坐标
      2) sendGraphData()   key:value 逗号分隔：
         smoothRoll, smoothPitch, PlateRoll, PlatePitch, MeasX, MeasY
    """

    def __init__(self, port: str, baud: int = 115200) -> None:
        import serial  # 延迟导入，未装 pyserial 时不影响其他模式
        self.ser = serial.Serial(port, baud, timeout=0.3)
        self.bad_lines = 0

    def close(self) -> None:
        try:
            self.ser.close()
        except Exception:
            pass

    def read_frame(self) -> Optional[Frame]:
        line = self.ser.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            return None
        frame = self._parse(line)
        if frame is None:
            self.bad_lines += 1
        return frame

    def _parse(self, line: str) -> Optional[Frame]:
        try:
            if ":" in line:
                kv = {}
                for part in line.split(","):
                    if ":" in part:
                        k, v = part.split(":", 1)
                        kv[k.strip().lower()] = float(v)
                x = kv.get("measx", kv.get("x", 0.0))
                y = kv.get("measy", kv.get("y", 0.0))
                roll = kv.get("smoothroll", kv.get("roll", 0.0))
                pitch = kv.get("smoothpitch", kv.get("pitch", 0.0))
                return Frame(t=time.time(), ir=[0.0] * 16,
                             x=x * SPACING_MM, y=y * SPACING_MM,
                             roll=roll, pitch=pitch,
                             note="仅姿态帧，无传感器数据")

            vals = [float(v) for v in line.split(",") if v.strip() != ""]
            if len(vals) >= 20:
                ir = vals[0:16]
                cx, cy, sx, sy = vals[16], vals[17], vals[18], vals[19]
                return Frame(t=time.time(), ir=ir,
                             x=cx * SPACING_MM, y=cy * SPACING_MM,
                             sx=sx * SPACING_MM, sy=sy * SPACING_MM,
                             valid=(max(ir) - min(ir)) >= BALL_THRESHOLD)
            if len(vals) >= 2:
                return Frame(t=time.time(), ir=[0.0] * 16,
                             x=vals[0] * SPACING_MM, y=vals[1] * SPACING_MM,
                             note="帧字段不足，仅解析位置")
        except (ValueError, IndexError):
            return None
        return None


# ---------------------------------------------------------------------------
# CSV 回放源
# ---------------------------------------------------------------------------

class CsvSource:
    """CSV 至少包含 t, x, y；可选 roll, pitch, ir0..ir15。按时间戳节奏回放。"""

    def __init__(self, path: str, loop: bool = True) -> None:
        self.path = path
        self.loop = loop
        self.rows: list[dict] = []
        with open(path, "r", encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    self.rows.append({
                        "t": float(row.get("t") or 0),
                        "x": float(row.get("x") or 0),
                        "y": float(row.get("y") or 0),
                        "roll": float(row.get("roll") or 0),
                        "pitch": float(row.get("pitch") or 0),
                        "ir": [float(row.get(f"ir{i}") or 0) for i in range(16)],
                    })
                except (TypeError, ValueError):
                    continue
        self.rows.sort(key=lambda r: r["t"])
        self.idx = 0
        self.t0_wall = time.time()
        self.t0_data = self.rows[0]["t"] if self.rows else 0.0

    def read_frame(self) -> Optional[Frame]:
        if not self.rows:
            return None
        elapsed = time.time() - self.t0_wall
        target = self.t0_data + elapsed
        row = self.rows[min(self.idx, len(self.rows) - 1)]
        while self.idx < len(self.rows) - 1 and self.rows[self.idx]["t"] <= target:
            self.idx += 1
            row = self.rows[self.idx]
        if self.idx >= len(self.rows) - 1:
            if self.loop:
                self.idx = 0
                self.t0_wall = time.time()
                self.t0_data = self.rows[0]["t"]
            else:
                return None
        return Frame(t=time.time(), ir=row["ir"], x=row["x"], y=row["y"],
                     roll=row["roll"], pitch=row["pitch"])


# ---------------------------------------------------------------------------
# 后台采集器
# ---------------------------------------------------------------------------

class DataReader:
    """后台线程持续采集，对外提供线程安全快照。"""

    def __init__(self, mode: str = "sim", port: Optional[str] = None,
                 baud: int = 115200, csv_path: Optional[str] = None,
                 maxlen: int = 7200) -> None:
        self.mode = mode
        self.port = port
        self.baud = baud
        self.csv_path = csv_path
        self.frames: deque[Frame] = deque(maxlen=maxlen)
        self.lock = threading.Lock()
        self.error: Optional[str] = None
        self.connected = False
        self.total = 0
        self.dropped = 0
        self.started_at = time.time()
        self._stop = threading.Event()

        # 同步创建数据源，让首屏立刻有内容；失败时把原因留给界面展示
        self._source = self._make_source()
        if self._source is not None and mode == "sim":
            self.frames.append(self._source.step(1.0 / FPS))

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _make_source(self):
        try:
            if self.mode == "serial":
                return SerialSource(self.port, self.baud)
            if self.mode == "csv":
                return CsvSource(self.csv_path)
            return SimulatedSource()
        except Exception as exc:  # 串口占用 / 文件不存在等
            self.error = f"数据源初始化失败：{exc}"
            return None

    def _run(self) -> None:
        src = self._source
        if src is None:
            return

        self.connected = True
        next_tick = time.time()
        while not self._stop.is_set():
            try:
                if self.mode == "sim":
                    next_tick += 1.0 / FPS
                    sleep = next_tick - time.time()
                    if sleep > 0:
                        time.sleep(sleep)
                    frame = src.step()
                else:
                    frame = src.read_frame()
                    if frame is None:
                        self.dropped += 1
                        time.sleep(0.02)
                        continue

                with self.lock:
                    self.frames.append(frame)
                    self.total += 1
            except Exception as exc:
                self.error = f"采集异常：{exc}"
                time.sleep(0.5)

    def snapshot(self) -> list[Frame]:
        with self.lock:
            return list(self.frames)

    def stats(self) -> dict:
        with self.lock:
            return {
                "total": self.total,
                "dropped": self.dropped,
                "buffered": len(self.frames),
                "error": self.error,
                "connected": self.connected,
                "uptime": time.time() - self.started_at,
            }

    def stop(self) -> None:
        self._stop.set()
