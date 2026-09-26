#!/usr/bin/env python3
"""Chess Clock: a two-player chess clock played with the bar's buttons and wheel.

A white or black king shows whose turn it is and slides across on every move,
with the time left in large digits and a bar that shrinks as it runs down.
Fischer increments, beeps in the last 10 seconds and an alarm on a flag.

    python app.py                        # BUSY Bar over USB (always 10.0.4.20)
    python app.py --host 127.0.0.1:8080  # emulator, or a Wi-Fi bar's IP
    python app.py --demo                 # plays short games by itself

Controls: wheel picks the time control; BACK and OK are the left and right
players' buttons (--swap flips them); START starts/pauses, hold it to reset.
Over Wi-Fi, pass the bar's API password with --token or BUSYBAR_TOKEN.
Run with --help for layouts, player names, and the other options.

Generated from the chessclock package by tools/build_gallery.py.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import logging
import math
import os
import struct
import sys
import termios
import time
import tty
import wave
import zlib
from dataclasses import dataclass
from enum import Enum
from typing import Callable


# ------------------------------ clock ---------------------------------------

LEFT, RIGHT = 0, 1


@dataclass(frozen=True)
class TimeControl:
    minutes: float
    increment: int = 0  # seconds added after each move (Fischer)

    @property
    def label(self) -> str:
        base = f"{self.minutes:g}"
        return f"{base}+{self.increment}" if self.increment else f"{base} min"


PRESETS: list[TimeControl] = [
    TimeControl(1, 0),
    TimeControl(2, 1),
    TimeControl(3, 0),
    TimeControl(3, 2),
    TimeControl(5, 0),
    TimeControl(5, 3),
    TimeControl(10, 0),
    TimeControl(10, 5),
    TimeControl(15, 10),
    TimeControl(30, 0),
    TimeControl(60, 0),
]


class Phase(Enum):
    READY = "ready"  # time control chosen, no clock running yet
    RUNNING = "running"
    PAUSED = "paused"
    FLAGGED = "flagged"  # someone ran out of time


class ChessClock:
    def __init__(
        self,
        control: TimeControl,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._now = now
        self.control = control
        start_ms = int(control.minutes * 60_000)
        self.remaining_ms = [start_ms, start_ms]
        self.moves = [0, 0]
        self.active: int | None = None
        self.flagged: int | None = None
        self.phase = Phase.READY
        self._since = self._now()

    def _settle(self) -> None:
        """Charge elapsed time to the active player and detect a flag."""
        t = self._now()
        if self.phase is not Phase.RUNNING or self.active is None:
            self._since = t
            return
        elapsed = int((t - self._since) * 1000)
        # Advance by exactly the whole ms charged, so the sub-ms remainder
        # carries over instead of being lost on every frame.
        self._since += elapsed / 1000
        left = self.remaining_ms[self.active] - elapsed
        if left <= 0:
            self.remaining_ms[self.active] = 0
            self.flagged = self.active
            self.phase = Phase.FLAGGED
        else:
            self.remaining_ms[self.active] = left

    def tick(self) -> None:
        self._settle()

    def press(self, player: int) -> None:
        """Player hits their button: ends their turn and starts the opponent's."""
        self._settle()
        other = 1 - player
        if self.phase is Phase.READY:
            # Like a physical clock: pressing your side starts the opponent's.
            self.active = other
            self.phase = Phase.RUNNING
        elif self.phase is Phase.RUNNING and self.active == player:
            self.moves[player] += 1
            self.remaining_ms[player] += self.control.increment * 1000
            self.active = other
        elif self.phase is Phase.PAUSED and self.active == player:
            # Finishing a move while paused resumes on the opponent's clock.
            self.moves[player] += 1
            self.remaining_ms[player] += self.control.increment * 1000
            self.active = other
            self.phase = Phase.RUNNING
        # Presses out of turn, or after a flag, are ignored.

    def toggle_pause(self) -> None:
        self._settle()
        if self.phase is Phase.RUNNING:
            self.phase = Phase.PAUSED
        elif self.phase is Phase.PAUSED:
            self.phase = Phase.RUNNING
        elif self.phase is Phase.READY:
            # Start with the left player to move.
            self.active = LEFT
            self.phase = Phase.RUNNING

    @property
    def full_moves(self) -> int:
        return min(self.moves)  # completed full moves (both sides played)

    @property
    def winner(self) -> int | None:
        """The side that still had time when the other flagged."""
        return None if self.flagged is None else 1 - self.flagged


def format_ms(ms: int) -> str:
    """m:ss (minutes past 59 stay minutes, to fit the display), s.t under 10 seconds."""
    if ms < 10_000:
        return f"{ms // 1000}.{(ms % 1000) // 100}"
    # Round down like the tenths below, so 0:10 gets its full second before 9.9.
    m, s = divmod(ms // 1000, 60)
    return f"{m}:{s:02}"


# ------------------------------ sprites -------------------------------------

# 16x16 king silhouette: cross, crown, body, base.
KING = [
    ".......##.......",
    "......####......",
    ".......##.......",
    "..###..##..###..",
    ".#####.##.#####.",
    ".##############.",
    ".##############.",
    "..############..",
    "...##########...",
    "....########....",
    "....########....",
    "...##########...",
    "....########....",
    "...##########...",
    "..############..",
    "..############..",
]
KING_SIZE = 16
COLLAR_ROW = 12  # drawn in the opposite shade to give both kings some detail


def _lit(x: int, y: int) -> bool:
    return 0 <= y < KING_SIZE and 0 <= x < KING_SIZE and KING[y][x] == "#"


def _edge(x: int, y: int) -> bool:
    return any(not _lit(x + dx, y + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))


def king_pixels(white: bool) -> list[list[tuple[int, int, int, int]]]:
    """White king: solid. Black king: outline only, since unlit LEDs are the 'black'."""
    on, off = (255, 255, 255, 255), (0, 0, 0, 0)

    def shade(x: int, y: int) -> bool:
        if not _lit(x, y):
            return False
        if _edge(x, y):
            return True
        return white != (y == COLLAR_ROW)

    return [[on if shade(x, y) else off for x in range(KING_SIZE)] for y in range(KING_SIZE)]


def encode_png(pixels: list[list[tuple[int, int, int, int]]]) -> bytes:
    h, w = len(pixels), len(pixels[0])
    raw = b"".join(b"\x00" + bytes(c for px in row for c in px) for row in pixels)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


KING_FILES = {True: "king_w.png", False: "king_b.png"}


def king_assets() -> dict[str, bytes]:
    """Filename -> PNG bytes, ready for assets_upload."""
    return {name: encode_png(king_pixels(white)) for white, name in KING_FILES.items()}


# ------------------------------ sounds --------------------------------------

RATE = 44_100  # the bar's firmware plays 44.1 kHz 16-bit mono PCM

TICK_FILE = "tick.wav"
FLAG_FILE = "flag.wav"


def _tone(freq: float, seconds: float, volume: float = 0.5) -> list[int]:
    n = int(RATE * seconds)
    fade = min(n // 2, int(RATE * 0.005))  # 5 ms ramps avoid clicks
    out = []
    for i in range(n):
        env = min(1.0, i / fade, (n - i) / fade) if fade else 1.0
        out.append(int(32767 * volume * env * math.sin(2 * math.pi * freq * i / RATE)))
    return out


def _wav(samples: list[int]) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return buf.getvalue()


def sound_assets() -> dict[str, bytes]:
    """Filename -> WAV bytes, ready for assets_upload."""
    gap = [0] * int(RATE * 0.08)
    flag = (_tone(880, 0.25) + gap + _tone(660, 0.25) + gap) * 2 + _tone(880, 0.5)
    return {TICK_FILE: _wav(_tone(1000, 0.08)), FLAG_FILE: _wav(flag)}


class BeepTracker:
    """Decides, frame by frame, which sound (if any) the clock state calls for.

    One tick per second while the running player is under `low_ms`, and the
    flag alarms once when someone runs out of time.
    """

    def __init__(self, low_ms: int) -> None:
        self.low_ms = low_ms
        self._last: tuple[int, int] | None = None  # (player, whole seconds left) last beeped
        self._flagged = False

    def update(self, clock: ChessClock | None) -> str | None:
        if clock is None or clock.phase is Phase.READY:
            self._last, self._flagged = None, False
            return None
        if clock.phase is Phase.FLAGGED:
            if self._flagged:
                return None
            self._flagged = True
            return FLAG_FILE
        if clock.phase is not Phase.RUNNING or clock.active is None:
            return None
        ms = clock.remaining_ms[clock.active]
        if ms >= self.low_ms:
            return None
        second = (clock.active, ms // 1000)
        if second == self._last:
            return None
        self._last = second
        return TICK_FILE


# ------------------------------ render --------------------------------------

FRONT_W, FRONT_H = 72, 16
HALF = FRONT_W // 2
BACK_W, BACK_H = 160, 80

WHITE = "#FFFFFFFF"
DIM = "#505050FF"
GREEN = "#00C040FF"
AMBER = "#FFA000FF"
RED = "#FF2020FF"

LOW_TIME_MS = 10_000

# Single layout geometry: king on the left, time centred in the rest, bar on the bottom row.
TIME_X = KING_SIZE + (FRONT_W - KING_SIZE) // 2
TIME_Y = 7
BAR_X = KING_SIZE + 2
BAR_W = FRONT_W - BAR_X
# Elements not in use are parked off the left edge rather than deleted: the bar keeps
# an app's elements by id, and parking left is the gallery's accepted way to hide one.
PARK_X = -200


def _text(id_, text, x, y, font, display="front", color=WHITE, align="center"):
    return {
        "id": id_,
        "type": "text",
        "text": text,
        "x": x,
        "y": y,
        "font": font,
        "color": color,
        "align": align,
        "display": display,
    }


def _rect(id_, x, y, w, h, color, fill="none"):
    return {
        "id": id_,
        "type": "rectangle",
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "fill": fill,
        "fill_colors": [color],
        "border_width": 0 if fill == "solid" else 1,
        "border_color": color,
        "display": "front",
    }


def _king(id_, player, x, y):
    return {"id": id_, "type": "image", "path": KING_FILES[player == LEFT], "x": x, "y": y, "display": "front"}


def render_setup(control: TimeControl, names: tuple[str, str], layout: str) -> list[dict]:
    if layout == "single":
        front = [
            _king("setup_w", LEFT, 0, 0),
            _king("setup_b", RIGHT, FRONT_W - KING_SIZE, 0),
            _text("setup", control.label, FRONT_W // 2, TIME_Y, "large", color=AMBER),
        ]
    else:
        front = [_text("setup", control.label, FRONT_W // 2, FRONT_H // 2, "bold", color=AMBER)]
    return front + [
        _text("b_title", "CHESS CLOCK", BACK_W // 2, 12, "bold", "back"),
        _text("b_tc", control.label, BACK_W // 2, 36, "large", "back"),
        _text("b_hint1", "Wheel: time control", BACK_W // 2, 60, "small", "back"),
        _text("b_hint2", f"START: {names[LEFT]} moves first", BACK_W // 2, 72, "small", "back"),
    ]


def _status(clock: ChessClock, names: tuple[str, str]) -> str:
    if (winner := clock.winner) is not None:
        return f"{names[winner]} wins on time"
    if clock.phase is Phase.PAUSED:
        return "Paused"
    if clock.active is None:
        return "Press to start"
    return f"{names[clock.active]} to move"


def _back(clock: ChessClock, names: tuple[str, str]) -> list[dict]:
    return [
        _text("b_tc", clock.control.label, BACK_W // 2, 10, "small", "back"),
        _text("b_status", _status(clock, names), BACK_W // 2, 30, "bold", "back"),
        _text(
            "b_times",
            f"{format_ms(clock.remaining_ms[LEFT])}   {format_ms(clock.remaining_ms[RIGHT])}",
            BACK_W // 2,
            52,
            "large",
            "back",
        ),
        _text("b_moves", f"Move {clock.full_moves + 1}", BACK_W // 2, 72, "small", "back"),
    ]


def _time_text(clock: ChessClock, player: int, blink_on: bool) -> str:
    if clock.flagged == player and not blink_on:
        return "FLAG"
    return format_ms(clock.remaining_ms[player])


def _time_color(clock: ChessClock, player: int) -> str:
    if clock.flagged == player:
        return RED
    if clock.phase is Phase.PAUSED:
        return AMBER
    if clock.active != player or clock.phase is not Phase.RUNNING:
        return DIM
    return RED if clock.remaining_ms[player] < LOW_TIME_MS else WHITE


def render_split(clock: ChessClock, names: tuple[str, str], blink_on: bool) -> list[dict]:
    els: list[dict] = []
    for player, cx in ((LEFT, HALF // 2), (RIGHT, HALF + HALF // 2)):
        if clock.winner == player:
            text, color = "WIN", GREEN
        else:
            text, color = _time_text(clock, player, blink_on), _time_color(clock, player)
        els.append(_text(f"t{player}", text, cx, FRONT_H // 2, "bold", color=color))

    # A frame around whoever's clock is ticking (or would tick on resume), or the winner.
    if (winner := clock.winner) is not None:
        els.append(_rect("frame", winner * HALF, 0, HALF, FRONT_H, GREEN))
    elif clock.active is not None:
        color = GREEN if clock.phase is Phase.RUNNING else AMBER
        els.append(_rect("frame", clock.active * HALF, 0, HALF, FRONT_H, color))
    return els + _back(clock, names)


def _bar(clock: ChessClock, player: int) -> dict:
    start = clock.control.minutes * 60_000
    frac = min(1.0, clock.remaining_ms[player] / start) if start else 0.0
    if clock.phase is Phase.READY:
        color = DIM
    elif clock.phase is Phase.PAUSED:
        color = AMBER
    elif clock.winner == player:
        color = GREEN
    else:
        color = GREEN if frac > 0.5 else AMBER if frac > 0.2 else RED
    return _rect("bar", BAR_X, FRONT_H - 1, max(1, round(BAR_W * frac)), 1, color, fill="solid")


def _out_text(clock: ChessClock, player: int) -> str:
    return "FLAG" if clock.flagged == player else format_ms(clock.remaining_ms[player])


def render_single(
    clock: ChessClock,
    names: tuple[str, str],
    blink_on: bool,
    out_player: int | None = None,
    progress: float = 1.0,
    celebrate: bool = False,
) -> list[dict]:
    """One clock for the side to move. While `progress` < 1, the previous side
    (`out_player`) slides out to the left as the new side slides in from the right.
    With `celebrate` after a flag, the winner is shown instead of the flagged side."""
    winner = clock.winner if celebrate else None
    if winner is not None:
        player, text, color = winner, "WINS", GREEN
    else:
        player = LEFT if clock.active is None else clock.active
        text, color = _time_text(clock, player, blink_on), _time_color(clock, player)
    out = -1 if out_player is None else out_player
    sliding = out >= 0 and out != player and progress < 1
    eased = 1 - (1 - progress) ** 2 if sliding else 1.0
    dx_in = round(FRONT_W * (1 - eased))

    els = [
        _king("king", player, dx_in, 0),
        _text("time", text, TIME_X + dx_in, TIME_Y, "extra_large", color=color),
        _bar(clock, player),
    ]
    if sliding:
        dx_out = -round(FRONT_W * eased)
        els += [
            _king("king_out", out, dx_out, 0),
            _text("time_out", _out_text(clock, out), TIME_X + dx_out, TIME_Y, "extra_large", color=DIM),
        ]
    else:
        els += [
            _king("king_out", RIGHT, PARK_X, 0),
            _text("time_out", " ", PARK_X, TIME_Y, "extra_large"),
        ]
    return els + _back(clock, names)


# ------------------------------ app -----------------------------------------

APP_NAME = "chess-clock"
FRAME_INTERVAL = 0.05
LONG_PRESS_S = 1.0
SLIDE_S = 0.3  # turn-switch animation length
WINNER_AFTER_S = 2.0  # how long FLAG blinks before the winner slides in
RECONNECT_S = 1.0
RETRY_S = 1.0  # wait after a refused draw, e.g., while a higher-priority app has the screen
DEMO_CONTROL = TimeControl(0.2)  # 12 s a side: reaches the red, tenths-of-a-second zone quickly
# Thinking time per move, White then Black: after five moves each White has 3 s left to Black's 5,
# and White flags on the next move.
DEMO_MOVE_S = (1.8, 1.4)

log = logging.getLogger("chessclock")


class ClockApp:
    """Owns the setup/game state and translates abstract actions into clock calls."""

    def __init__(self, names: tuple[str, str], preset: int, layout: str = "single") -> None:
        self.names = names
        self.preset = preset % len(PRESETS)
        self.layout = layout
        self.clock: ChessClock | None = None
        self.quit = asyncio.Event()
        # Turn-switch animation: who was shown before, and since when the new side is.
        self._shown: int | None = None
        self._out: int | None = None
        self._switched_at = 0.0
        self._flagged_at: float | None = None

    def _game(self) -> ChessClock:
        """The running game, starting one with the chosen preset if needed."""
        clock = self.clock
        if clock is None:
            clock = self.clock = ChessClock(PRESETS[self.preset])
        return clock

    def player(self, player: int) -> None:
        self._game().press(player)

    def start_or_pause(self) -> None:
        clock = self._game()
        if clock.phase is Phase.FLAGGED:
            self.reset()
        else:
            clock.toggle_pause()

    def reset(self) -> None:
        self.clock = None

    def wheel(self, delta: int) -> None:
        if self.clock is None:
            self.preset = (self.preset + delta) % len(PRESETS)

    def frame(self) -> list[dict]:
        clock = self.clock
        if clock is None:
            self._shown = self._flagged_at = None
            return render_setup(PRESETS[self.preset], self.names, self.layout)
        clock.tick()
        now = time.monotonic()
        blink_on = int(now * 2) % 2 == 0
        if self.layout == "split":
            return render_split(clock, self.names, blink_on)

        celebrate = False
        if clock.phase is Phase.FLAGGED:
            if self._flagged_at is None:
                self._flagged_at = now
            celebrate = now - self._flagged_at >= WINNER_AFTER_S
        else:
            self._flagged_at = None
        shown = clock.winner if celebrate else clock.active
        shown = LEFT if shown is None else shown
        if shown != self._shown:
            self._out = self._shown
            self._shown = shown
            self._switched_at = now
        progress = min(1.0, (now - self._switched_at) / SLIDE_S)
        return render_single(clock, self.names, blink_on, self._out, progress, celebrate)


async def device_input(bar, app: ClockApp, left_btn: str, right_btn: str) -> None:
    from busylib.features import ButtonEvent, EncoderEvent, input_events

    start_down: float | None = None
    lost = False
    while not app.quit.is_set():
        try:
            async for message in bar.stream_status_ws():
                if lost:
                    log.info("input stream back")
                    lost = False
                if not isinstance(message, dict):
                    continue
                for ev in input_events(message):
                    if isinstance(ev, EncoderEvent):
                        app.wheel(ev.delta)
                    elif isinstance(ev, ButtonEvent):
                        if ev.button == "start":
                            # Short press pauses/resumes, long press resets.
                            if ev.is_press:
                                start_down = time.monotonic()
                            elif start_down is not None:
                                held = time.monotonic() - start_down
                                start_down = None
                                if held >= LONG_PRESS_S:
                                    app.reset()
                                else:
                                    app.start_or_pause()
                        elif ev.is_press and ev.button == left_btn:
                            app.player(LEFT)
                        elif ev.is_press and ev.button == right_btn:
                            app.player(RIGHT)
        except Exception as exc:  # the stream drops if the bar reboots or USB blips
            # Warn once per outage, not on every retry.
            (log.debug if lost else log.warning)("input stream lost (%s), reconnecting", exc)
            lost = True
        # Wait before reconnecting, also when the bar closed the stream cleanly.
        if not app.quit.is_set():
            await asyncio.sleep(RECONNECT_S)


@contextlib.contextmanager
def keyboard(app: ClockApp):
    """Optional terminal controls: a/l players, space pause, r reset, +/- wheel, q quit."""
    if not sys.stdin.isatty():
        yield
        return
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    keys = {
        "a": lambda: app.player(LEFT),
        "l": lambda: app.player(RIGHT),
        " ": app.start_or_pause,
        "r": app.reset,
        "+": lambda: app.wheel(1),
        "=": lambda: app.wheel(1),
        "-": lambda: app.wheel(-1),
        "q": app.quit.set,
    }

    def on_key() -> None:
        action = keys.get(os.read(fd, 1).decode(errors="ignore").lower())
        if action:
            action()

    loop = asyncio.get_running_loop()
    loop.add_reader(fd, on_key)
    try:
        yield
    finally:
        loop.remove_reader(fd)
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def terminal_line(elements: list[dict]) -> str:
    kings = {"king_w.png": "♔", "king_b.png": "♚"}
    front = [
        kings.get(e.get("path", ""), "") + e.get("text", "")
        for e in elements
        if e["display"] == "front" and e["type"] in ("text", "image") and -8 < e["x"] < 72
    ]
    back = [e["text"] for e in elements if e["display"] == "back" and e["type"] == "text"]
    return f"[ {'  |  '.join(front)} ]   {' · '.join(back)}"


async def render_loop(app: ClockApp, draw, play=None) -> None:
    last: list[dict] | None = None
    parked: dict[str, dict] = {}
    screen_taken = False
    beeps = BeepTracker(LOW_TIME_MS)
    pending: set[asyncio.Task] = set()
    while not app.quit.is_set():
        elements = app.frame()
        if play and (sound := beeps.update(app.clock)):
            # Fire and forget, so a slow audio request never stalls the clock display.
            task = asyncio.create_task(play(sound))
            pending.add(task)
            task.add_done_callback(pending.discard)

        # The bar keeps an app's elements by id until they are deleted, so an element
        # left out of a frame would stay on screen. Every id ever drawn is therefore
        # sent in every frame, with the ones not in use parked off the left edge.
        ids = {e["id"] for e in elements}
        frame = elements + [e for i, e in parked.items() if i not in ids]
        parked.update({e["id"]: {**e, "x": PARK_X} for e in elements})

        if frame != last:
            try:
                await draw(frame)
                last = frame
                if screen_taken:
                    log.info("got the screen back")
                    screen_taken = False
            except Exception as exc:
                if getattr(exc, "status_code", None) == 409:
                    # A higher-priority app owns the screen: normal, keep the clock running.
                    if not screen_taken:
                        log.info("a higher-priority app has the screen, waiting for it")
                        screen_taken = True
                else:
                    log.warning("draw failed: %s", exc)
                await asyncio.sleep(RETRY_S)
                continue
        await asyncio.sleep(FRAME_INTERVAL)


async def demo(app: ClockApp) -> None:
    """Plays short games by itself, to show the clock off (and record previews)."""
    while not app.quit.is_set():
        clock = app.clock = ChessClock(DEMO_CONTROL)
        clock.toggle_pause()
        while clock.phase is not Phase.FLAGGED and not app.quit.is_set():
            await asyncio.sleep(DEMO_MOVE_S[LEFT if clock.active is None else clock.active])
            if clock.active is not None and clock.phase is Phase.RUNNING:
                app.player(clock.active)
        await asyncio.sleep(WINNER_AFTER_S + 3)  # FLAG, then the winner


async def run(args: argparse.Namespace) -> None:
    app = ClockApp((args.left_name, args.right_name), args.preset, args.layout)
    left_btn, right_btn = ("ok", "back") if args.swap else ("back", "ok")

    with keyboard(app):
        if args.dry_run:
            async def draw_terminal(elements):
                print("\r\033[K" + terminal_line(elements), end="", flush=True)

            async def bell(_sound):
                print("\a", end="", flush=True)

            # Created before the render loop so the very first frame is already the game.
            demo_task = asyncio.create_task(demo(app)) if args.demo else None
            await render_loop(app, draw_terminal, None if args.no_beep else bell)
            if demo_task:
                demo_task.cancel()
            print()
            return

        from busylib import AsyncBusyBar, types

        # Entered without `as`: busylib types __aenter__ as its base class,
        # which would hide the client's methods from type checkers.
        bar = AsyncBusyBar(args.host, token=args.token)
        async with bar:
            assets = king_assets() | ({} if args.no_beep else sound_assets())
            for filename, data in assets.items():
                await bar.assets_upload(application_name=APP_NAME, filename=filename, data=data)

            async def draw_bar(elements):
                payload = types.DisplayElements(application_name=APP_NAME, elements=elements)
                await bar.display_draw(payload)

            if args.test:
                # Smoke test: draw the setup screen once and exit, leaving it up.
                await draw_bar(app.frame())
                return

            async def play(sound):
                try:
                    await bar.audio_play(path=sound, application_name=APP_NAME)
                except Exception as exc:
                    log.warning("beep failed: %s", exc)

            # The demo starts first, so the very first frame is already the game.
            tasks = [asyncio.create_task(demo(app))] if args.demo else []
            tasks += [
                asyncio.create_task(render_loop(app, draw_bar, None if args.no_beep else play)),
                asyncio.create_task(device_input(bar, app, left_btn, right_btn)),
            ]
            try:
                await app.quit.wait()
            finally:
                for t in tasks:
                    t.cancel()
                with contextlib.suppress(Exception):
                    await bar.display_clear(application_name=APP_NAME)


def main() -> None:
    p = argparse.ArgumentParser(description="Chess clock for the BUSY Bar")
    p.add_argument("--host", "--addr", dest="host", default=os.environ.get("BUSYBAR_HOST", "10.0.4.20"),
                   help="bar address[:port]: 10.0.4.20 over USB (default), its IP over Wi-Fi, "
                        "or 127.0.0.1:8080 for the emulator")
    p.add_argument("--token", default=os.environ.get("BUSYBAR_TOKEN"),
                   help="API password, needed over Wi-Fi (or set BUSYBAR_TOKEN)")
    p.add_argument("--preset", type=int, default=5,
                   help="initial time control index: " +
                        ", ".join(f"{i}={tc.label}" for i, tc in enumerate(PRESETS)))
    p.add_argument("--swap", action="store_true",
                   help="swap which physical button (BACK/OK) belongs to each side")
    p.add_argument("--layout", choices=["single", "split"], default="single",
                   help="single: one clock with a king for the side to move (default); "
                        "split: both clocks side by side")
    p.add_argument("--no-beep", action="store_true",
                   help="silence the last-10-seconds beeps and the flag alarm")
    p.add_argument("--left-name", default="White")
    p.add_argument("--right-name", default="Black")
    p.add_argument("--dry-run", action="store_true",
                   help="no device: show the screens in the terminal, drive with the keyboard")
    p.add_argument("--demo", action="store_true",
                   help="play short games by itself, e.g. to show the clock off")
    p.add_argument("--test", action="store_true",
                   help="draw the setup screen once and exit")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    if not args.verbose:
        # Parked and sliding elements sit off the display on purpose.
        logging.getLogger("busylib.client.display").setLevel(logging.ERROR)
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
