"""Animate one episode of a policy on QuadEnv: MuJoCo view plus command and attitude plots."""
import os
import time
from collections.abc import Callable, Iterable

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, SubplotSpec
from matplotlib.lines import Line2D
from PIL import Image

from configs import TASKS, Config, load_config, load_env
from envs.evaluation import SETTLE_WINDOW, Policy
from envs.quad_env import QuadEnv
from envs.quadrotor import SAT_COLLECTIVE, SAT_ROLL_PITCH, SAT_YAW, G

ARROW_PER_HOVER = 0.12  # thrust arrow length [m] at hover thrust (m*g)
VIEW_SIZE = {True: (400, 800), False: (500, 500)}  # (height, width) px, with / without plots
CAMERA_AZIMUTH = 205.0    # deg; 180 looks along the roll axis
CAMERA_ELEVATION = -12.0  # deg, looking slightly down
FRAME_DT = 0.02  # s per frame (50 fps)
BASELINES = ["zero", "hover", "random"] + [f"scripted_{task}" for task in TASKS]
MODEL_FILES = {"ppo_": ".zip", "tdmpc2_": ".pt"}  # name prefix -> file extension


def trained_models() -> list[str]:
    """Names of the trained models in models/, e.g. ppo_flip, ppo_flip_v1, tdmpc2_flip_prev."""
    if not os.path.isdir("models"):
        return []
    names = []
    for f in sorted(os.listdir("models")):
        stem, ext = os.path.splitext(f)
        if any(stem.startswith(p) and ext == e for p, e in MODEL_FILES.items()) and not stem.endswith(".tmp"):
            names.append(stem)
    return names


def task_of(policy: str) -> str:
    """The first task named in the policy (ppo_flip_v1 -> flip); open-loop baselines run in the hover env."""
    return next((part for part in policy.split("_") if part in TASKS), "hover")


def env_config_for(policy: str) -> Config:
    """Trained models run in the env they were trained in (saved with the model); baselines use env.yaml."""
    if policy.startswith("tdmpc2_"):
        return Config(torch.load(f"models/{policy}.pt", weights_only=False)["env_cfg"])
    if policy.startswith("ppo_"):
        from ppo.train_ppo import saved_env_config
        return saved_env_config(policy)
    return load_env(task_of(policy))


# Config keys that define the drone
DRONE_KEYS = ("dynamics", "latency", "sensors", "randomisation")
NOMINAL_DRAWS = {"hover_thrust": [1.0, 1.0], "thrust_curve": [0.0, 0.0], "inertia": [1.0, 1.0],
                 "motor_tau": [1.0, 1.0], "drag_xy": [1.0, 1.0], "drag_z": [1.0, 1.0]}


def env_config_with_drone(policy: str, drone_from: str) -> Config:
    """The policy's env, flying another model's drone (dynamics, latency, sensors, randomisation)."""
    cfg = env_config_for(policy)
    other = env_config_for(drone_from)
    if cfg.task != other.task:
        raise ValueError(f"{policy} is a {cfg.task} policy, {drone_from} a {other.task} one")
    dr = other.randomisation
    if dr is None and cfg.randomisation is not None:  # policy expects param inputs: draw them at nominal
        dr = NOMINAL_DRAWS
    return Config({**cfg, **{k: other[k] for k in DRONE_KEYS}, "randomisation": dr})


def scripted_hover_policy(env: QuadEnv, scripted: Config) -> Policy:
    """Cascaded PD hover controller."""
    def pd(obs: np.ndarray, t: int) -> np.ndarray:
        q = env.quad
        a_des = -scripted.pd_kp * (q.p - env.target) - scripted.pd_kd * q.v + np.array([0, 0, G])
        z_des = a_des / np.linalg.norm(a_des)
        rates = scripted.att_gain * np.cross([0, 0, 1], q.R.T @ z_des)
        thrust = a_des @ q.R[:, 2]
        a0 = (thrust - G) / (q.max_thrust_acc - G) if thrust >= G else thrust / G - 1.0
        return np.clip(np.array([a0, *(rates / q.max_rate)]), -1, 1)
    return pd


def scripted_flip_policy(env: QuadEnv, scripted: Config) -> Policy:
    """Climb, roll until one revolution, then recover with the hover PD controller."""
    pd = scripted_hover_policy(env, scripted)
    climb_steps = round(scripted.flip_climb_time / env.policy_dt)

    def flip(obs: np.ndarray, t: int) -> np.ndarray:
        if t < climb_steps:
            return np.array([1.0, 0, 0, 0])
        if not env.flip_done:
            return np.array([scripted.flip_thrust, 1.0, 0, 0])
        return pd(obs, t)
    return flip


def make_policy(name: str, env: QuadEnv, scripted: Config, plan: bool = True) -> Policy:
    """scripted: the `scripted` section of configs/animate_rollout.yaml (gains of the hand-written controllers)."""
    if name == "zero":
        return lambda obs, t: np.array([-1.0, 0, 0, 0])          # thrust off
    if name == "hover":
        return lambda obs, t: np.array([0.0, 0, 0, 0])           # thrust = g
    if name == "random":
        return lambda obs, t: env.action_space.sample()
    if name.startswith("ppo_"):
        from stable_baselines3 import PPO
        model = PPO.load(f"models/{name}")
        n = model.observation_space.shape[0]  # type: ignore[index]  # drops param inputs it wasn't trained with
        return lambda obs, t: model.predict(obs[:n], deterministic=True)[0]
    if name.startswith("tdmpc2_"):
        from tdmpc2.agent import TDMPC2
        agent, _ = TDMPC2.load(f"models/{name}.pt")
        return lambda obs, t: agent.act(obs[:agent.obs_dim], t0=t == 0, eval_mode=True, plan=plan)
    if name == "scripted_hover":
        return scripted_hover_policy(env, scripted)
    return scripted_flip_policy(env, scripted)


def rollout(env: QuadEnv, policy: Policy, seed: int) -> dict:
    """Log the state at t = 0 and after every step; index i is time i * policy_dt."""
    torch.manual_seed(seed)
    np.random.seed(seed)  # MPPI samples
    obs, _ = env.reset(seed=seed, options={"exact": True})
    log = {"p": [], "q": [], "R": [], "r": [], "dist": [], "revs": [], "done": [], "thrust": [],
           "thrust_cmd": [], "rate_cmd": [], "w": [], "speed": [], "mass": env.quad.m, "task": env.task,
           "action_change": [],  # sum |a - previous a|
           "saturation": [],  # Quadrotor.saturation
           "infer_ms": []}  # ms per policy call
    total = 0.0

    def record():
        q = env.quad
        log["p"].append(q.p.copy())
        log["q"].append(q.q.copy())
        log["R"].append(q.R.copy())
        log["r"].append(total)
        log["dist"].append(np.linalg.norm(q.p - env.target))
        log["revs"].append(env.roll_progress / (2 * np.pi))
        log["done"].append(env.flip_done)
        log["thrust"].append(q.m * q.thrust_acc)  # N, after motor lag
        log["w"].append(q.w.copy())
        log["speed"].append(np.linalg.norm(q.v))

    record()
    for t in range(env.max_steps):
        start = time.perf_counter()
        action = policy(obs, t)
        log["infer_ms"].append(1000 * (time.perf_counter() - start))
        action = np.clip(action, -1, 1)
        log["action_change"].append(np.abs(action - env.prev_action).sum())
        thrust_cmd, rate_cmd = env.scale_action(action)
        log["thrust_cmd"].append(env.quad.m * thrust_cmd)
        log["rate_cmd"].append(rate_cmd)
        obs, r, term, trunc, info = env.step(action)
        total += r
        log["saturation"].append(env.quad.saturation)
        record()
        if term or trunc:
            break
    log["crashed"] = info["crashed"]
    # Repeat the last command so commands line up with states
    log["thrust_cmd"].append(log["thrust_cmd"][-1])
    log["rate_cmd"].append(log["rate_cmd"][-1])
    return log


def print_summary(log: dict, env: QuadEnv):
    """How the episode ended, how still the drone is at the end, and the policy's inference time."""
    flip = f"  flip_done={env.flip_done}" if env.task == "flip" else ""
    print(f"steps={len(log['p']) - 1}  total reward={log['r'][-1]:.2f}  final dist={log['dist'][-1]:.3f}{flip}  "
          f"final upright={env.quad.R[2, 2]:.2f}  final z={env.quad.p[2]:.2f}  crashed={log['crashed']}  "
          f"mean |a - prev a|={np.mean(log['action_change']):.2f}")
    if env.params:  # domain randomisation
        p = env.params
        print(f"dynamics: hover thrust x{p['hover_thrust']:.2f}  thrust curve k={p['thrust_curve']:.2f}  "
              f"inertia x{np.round(p['inertia'], 2)}  motor lag x{p['motor_tau']:.2f}  drag x{np.round(p['drag'], 2)}  "
              f"latency {1000 * env.delay * env.quad.dt:.0f} ms")
    tail = max(1, round(SETTLE_WINDOW / env.policy_dt))
    print(f"last {SETTLE_WINDOW} s: mean dist={np.mean(log['dist'][-tail:]):.3f}  "
          f"mean speed={np.mean(log['speed'][-tail:]):.2f}  final speed={log['speed'][-1]:.2f}")
    saturation = np.array(log["saturation"])
    print(f"motor limits, share of steps: yaw reduced {100 * np.mean(saturation == SAT_YAW):.0f}%, "
          f"collective changed {100 * np.mean(saturation == SAT_COLLECTIVE):.0f}%, "
          f"roll/pitch reduced {100 * np.mean(saturation == SAT_ROLL_PITCH):.0f}%")
    # Skip the first call (warm-up)
    infer_ms = log["infer_ms"]
    steady = np.array(infer_ms[1:] or infer_ms)
    print(f"inference time per step: median={np.median(steady):.2f} ms  p99={np.percentile(steady, 99):.2f} ms  "
          f"first call={infer_ms[0]:.1f} ms  (control period {1000 * env.policy_dt:.0f} ms)")


def _camera(model: mujoco.MjModel, P: np.ndarray) -> mujoco.MjvCamera:
    """Fixed camera that keeps the whole trajectory P in view."""
    camera = mujoco.MjvCamera()
    camera.lookat = (P.min(0) + P.max(0)) / 2
    half = max(np.ptp(P, 0).max() / 2 + 0.25, 0.6)
    camera.distance = half / np.tan(np.radians(model.vis.global_.fovy) / 2)
    camera.azimuth, camera.elevation = CAMERA_AZIMUTH, CAMERA_ELEVATION
    return camera


def _add_connector(scene: mujoco.MjvScene, kind: mujoco.mjtGeom, width: float, start: np.ndarray, end: np.ndarray,
                   rgba: tuple[float, float, float, float]):
    """Add a line-like geom (the trail, the thrust arrow) to a scene that is about to be rendered."""
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, kind, np.zeros(3), np.zeros(3), np.eye(3).flatten(), np.array(rgba, np.float32))
    mujoco.mjv_connector(geom, kind, width, start, end)
    scene.ngeom += 1


def render_frames(log: dict, model: mujoco.MjModel, height: int, width: int) -> list[np.ndarray]:
    """Render each logged state in MuJoCo, with the trail and thrust arrow."""
    P = np.array(log["p"])
    data = mujoco.MjData(model)
    camera = _camera(model, P)
    hover_force = log["mass"] * G
    renderer = mujoco.Renderer(model, height, width)
    frames = []
    for i, (p, q) in enumerate(zip(P, log["q"], strict=True)):
        data.qpos[:3], data.qpos[3:7] = p, q
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera)
        for start, end in zip(P[:i], P[1:i + 1], strict=True):
            _add_connector(renderer.scene, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.006, start, end, (0.25, 0.3, 0.4, 0.6))
        tip = p + log["R"][i] @ [0, 0, ARROW_PER_HOVER * log["thrust"][i] / hover_force]
        _add_connector(renderer.scene, mujoco.mjtGeom.mjGEOM_ARROW, 0.012, p, tip, (0.1, 0.35, 0.9, 1.0))
        frames.append(renderer.render())
    renderer.close()  # explicitly: EGL errors if freed at exit
    return frames


def _setup_view(height: int, width: int, plots: bool) -> tuple[Figure, Axes, GridSpec | None]:
    """Figure and axes for the MuJoCo view, plus a grid for the plots (None without plots)."""
    if plots:
        fig = plt.figure(figsize=(9, 11))
        gs = fig.add_gridspec(5, 2, height_ratios=[4, 1, 1, 1, 1], hspace=0.25, wspace=0.3, top=0.98)
        ax = fig.add_subplot(gs[0, :])
    else:
        fig, gs = plt.figure(figsize=(width / 100, height / 100), dpi=100), None
        ax = fig.add_axes((0, 0, 1, 1))
    ax.axis("off")
    return fig, ax, gs


def _plot_series(log: dict, P: np.ndarray) -> dict[str, np.ndarray]:
    """The time series shown under the MuJoCo view, as arrays indexed by time step."""
    series = {"thrust_cmd": np.array(log["thrust_cmd"]), "thrust": np.array(log["thrust"]),
              "rate_cmd": np.array(log["rate_cmd"]), "w": np.array(log["w"]), "z": P[:, 2]}
    # ZYX Euler angles, unwrapped so a flip reads 0 -> 360 deg
    Rs = np.array(log["R"])
    euler = np.stack([np.arctan2(Rs[:, 2, 1], Rs[:, 2, 2]),
                      -np.arcsin(np.clip(Rs[:, 2, 0], -1, 1)),
                      np.arctan2(Rs[:, 1, 0], Rs[:, 0, 0])], 1)
    series["angle"] = np.degrees(np.unwrap(euler, axis=0))
    return series


def _add_panels(fig: Figure, gs: GridSpec, series: dict[str, np.ndarray], T: np.ndarray,
                hover_force: float) -> list[tuple[Line2D, str, int | None]]:
    """Plots under the MuJoCo view; returns [(line, series, column)] to update each frame."""
    # (grid cell, y label, colour, [(series, column, style, legend label)])
    panels: list[tuple[SubplotSpec, str, str, list[tuple[str, int | None, str, str | None]]]] = [
        (gs[1, 0], "thrust [N]", "b", [("thrust_cmd", None, "--", "cmd"), ("thrust", None, "-", "actual")]),
        (gs[1, 1], "height z [m]", "k", [("z", None, "-", None)])]
    for k, (name, col) in enumerate(zip(("roll", "pitch", "yaw"), ("C3", "C2", "C0"), strict=True)):
        panels += [(gs[k + 2, 0], f"{name} rate [rad/s]", col,
                    [("rate_cmd", k, "--", "cmd"), ("w", k, "-", "actual")]),
                   (gs[k + 2, 1], f"{name} [deg]", col, [("angle", k, "-", None)])]
    lines = []
    for cell, ylabel, col, curves in panels:
        a = fig.add_subplot(cell)
        ys = []
        for key, c, style, label in curves:
            # Commands are held over each step
            draw = "steps-post" if style == "--" else "default"
            lines.append((a.plot([], [], style, color=col, lw=1 if style == "--" else 1.5, label=label,
                                 drawstyle=draw)[0], key, c))
            ys.append(series[key] if c is None else series[key][:, c])
        lo, hi = min(y.min() for y in ys), max(y.max() for y in ys)
        pad = 0.1 * (hi - lo) + 1e-3
        a.set_xlim(0, T[-1])
        a.set_ylim(lo - pad, hi + pad)
        a.set_ylabel(ylabel, fontsize=8)
        a.grid(alpha=0.3)
        a.tick_params(labelsize=7)
        if "yaw" in ylabel:
            a.set_xlabel("time [s]")
        else:
            a.tick_params(labelbottom=False)
        if "thrust" in ylabel:
            a.axhline(hover_force, color="gray", lw=0.5)
            a.legend(loc="upper right", fontsize=7, ncol=2)
    return lines


def _save_gif(fig: Figure, update: Callable[[int], Iterable[Artist]], frames: list[int], path: str):
    """Save a GIF with one octree palette and no dithering (about 20x smaller)."""
    images = []
    for i in frames:
        update(i)
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())  # type: ignore[attr-defined]  # Agg canvas
        images.append(Image.fromarray(rgba).convert("RGB"))
    palette = images[-1].quantize(colors=256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE)
    images = [image.quantize(palette=palette, dither=Image.Dither.NONE) for image in images]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:], duration=round(FRAME_DT * 1000), loop=0)
    print(f"saved {path}")


def _play(fig: Figure, update: Callable[[int], Iterable[Artist]], n_states: int, policy_dt: float, start_pause: float,
          save: str | None) -> FuncAnimation:
    """Animate in real time (sim time = wall-clock time), after saving to `save` (a path) if given."""
    if save:
        hold = [0] * round(start_pause / FRAME_DT)  # hold t = 0
        _save_gif(fig, update, hold + list(range(0, n_states, max(1, round(FRAME_DT / policy_dt)))), save)
    end = (n_states - 1) * policy_dt

    def wall_clock_frames():
        """State index at the current wall-clock time; skips frames rather than playing slowly."""
        start = time.perf_counter() + start_pause
        while True:
            t = time.perf_counter() - start
            yield min(max(0, int(t / policy_dt)), n_states - 1)
            if t >= end:
                return

    anim = FuncAnimation(fig, update, frames=wall_clock_frames, interval=FRAME_DT * 1000, blit=True,
                         cache_frame_data=False)
    plt.show()
    return anim  # FuncAnimation stops if it is garbage collected


def animate(log: dict, model: mujoco.MjModel, policy_dt: float, save: str | None = None, start_pause: float = 0.0,
            plots: bool = True):
    """start_pause: s to hold t = 0; plots=False draws the MuJoCo view only."""
    P = np.array(log["p"])
    T = np.arange(len(P)) * policy_dt
    hover_force = log["mass"] * G
    height, width = VIEW_SIZE[plots]
    frames = render_frames(log, model, height, width)
    fig, ax, gs = _setup_view(height, width, plots)
    image = ax.imshow(frames[0])
    text = ax.text(0.01, 0.97, "", transform=ax.transAxes, va="top", fontsize=9)
    series = _plot_series(log, P)
    lines = _add_panels(fig, gs, series, T, hover_force) if gs is not None else []

    def update(i: int) -> tuple[Artist, ...]:
        image.set_data(frames[i])
        if plots:
            F = log["thrust"][i]
            flip = f"  revs={log['revs'][i]:.2f}  flip_done={log['done'][i]}" if log["task"] == "flip" else ""
            text.set_text(f"t={i * policy_dt:.2f}s  dist to target={log['dist'][i]:.2f} m{flip}\n"
                          f"thrust={F:.1f} N ({F / hover_force:.2f}x hover)  reward={log['r'][i]:.1f}")
        for line, key, col in lines:
            y = series[key][: i + 1]
            line.set_data(T[: i + 1], y if col is None else y[:, col])
        # Changing artists, for blitting
        return image, text, *(line for line, _, _ in lines)

    _play(fig, update, len(P), policy_dt, start_pause, save)


if __name__ == "__main__":
    cfg = load_config("animate_rollout")
    if cfg.policy not in BASELINES + trained_models():
        raise ValueError(f"policy must be a baseline {BASELINES} or a model in models/ {trained_models()}, "
                         f"got {cfg.policy!r}")
    if cfg.drone_from is None:
        env = QuadEnv(cfg=env_config_for(cfg.policy))
    else:
        if cfg.drone_from not in BASELINES + trained_models():
            raise ValueError(f"drone_from must be a model in models/ {trained_models()}, got {cfg.drone_from!r}")
        env = QuadEnv(cfg=env_config_with_drone(cfg.policy, cfg.drone_from))
    log = rollout(env, make_policy(cfg.policy, env, plan=cfg.plan, scripted=cfg.scripted), cfg.seed)
    print_summary(log, env)
    animate(log, env.quad.model, env.policy_dt, cfg.save, cfg.start_pause, cfg.plots)
