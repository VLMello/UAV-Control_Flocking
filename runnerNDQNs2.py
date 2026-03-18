"""
runnerDrawTrials_onlyDQN_with_MA.py

Same as the previous "only DQN" runner, but with a moving-average smoother added
for each DQN's average-distance trace.

Edit DQN_MODEL_FILES and DQN_LABELS at the top to configure which models to run.
"""

import os
import numpy as np
import math
import pickle
import time
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# -------------------------
# Environment params (tweak if needed)
# -------------------------
SIZE = 40
Q_TABLE_BASE_SIZE = 29   # must match training normalization
SHOW_SIZE = 800
HM_EPISODES = 2000       # number of episodes to average across
STEPS_PER_EPISODE = 300  # max timesteps per episode
SAFE_DISTANCE_LOW = 5
SAFE_DISTANCE_HIGH = 9

HEADING_MAX = 90
ROLL_MAX = 5
CHOICES = 3

SAFE_PLACE_REWARD = 0
ENEMY_PENALTY = 500

SHOW_EVERY = 200  # prints and progress every this many episodes
SHOW_ROUNDED_POSITION = False

ZOOM = int(SHOW_SIZE / SIZE)

# -------------------------
# Moving average config
# -------------------------
MOVING_AVERAGE_WINDOW = 21   # must be odd for symmetric smoothing; set to 1 to disable
PLOT_RAW = True              # whether to plot raw average curves (thin, faint)
# -------------------------
# DQN models configuration (edit here)
# -------------------------
# Put the exact paths to the model .pth files you want to compare.
# Example:
# DQN_MODEL_FILES = ["dqn_planes_1768713482.pth", "other_model.pth"]
DQN_MODEL_FILES = [
    "dqn_planes_1772477044.pth"
    # add more model files here...
]

# Optional friendly labels for each DQN model (same order). If empty or shorter,
# labels will default to "DQN #1", "DQN #2", ...
DQN_LABELS = [
    "DQN-Com Z6",
    "DQN-Sem Z6"
    # add more labels here...
]

COMMANDS_SEQUENCE_FILE = "commands-leader-1762474566.pickle"  # or None

SAVE_RESULTS_CSV = False
OUT_CSV_PREFIX = "dqn_trace"

# -------------------------
# Imports for DQN (attempt)
# -------------------------
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available — DQNs will fall back to random actions.")

# -------------------------
# Minimal environment (copied / adapted)
# -------------------------
class Plane:
    def __init__(self, x=None, y=None):
        if x is None:
            x = np.random.randint(0, SIZE)
        if y is None:
            y = np.random.randint(0, SIZE)
        self.x = float(x)
        self.y = float(y)
        self.speed = 1
        self.heading = 0
        self.roll = 0

    def step(self, phi=False):
        if phi is not False:
            self.roll = max(min(self.roll + phi, 30), -30)

        if self.roll == 30:
            self.heading = (self.heading + 2) % HEADING_MAX
        if self.roll == 15:
            self.heading = (self.heading + 1) % HEADING_MAX
        elif self.roll == -15:
            self.heading = (self.heading - 1) % HEADING_MAX
        elif self.roll == -30:
            self.heading = (self.heading - 2) % HEADING_MAX

        self.x += self.speed * np.cos(np.deg2rad(self.heading * 360 / HEADING_MAX))
        self.y += self.speed * np.sin(np.deg2rad(self.heading * 360 / HEADING_MAX))

        # clamp to grid
        self.x = min(max(self.x, 0.0), SIZE - 1.0)
        self.y = min(max(self.y, 0.0), SIZE - 1.0)

    def action(self, choice=None):
        if choice == 0:
            self.step(-15)
        elif choice == 1:
            self.step(0)
        elif choice == 2:
            self.step(15)
        else:
            self.step(np.random.choice([-15, 0, 15]))

    def get_discrete_roll(self):
        if self.roll == -30:
            return 0
        elif self.roll == -15:
            return 1
        elif self.roll == 0:
            return 2
        elif self.roll == 15:
            return 3
        elif self.roll == 30:
            return 4
        else:
            return 2

def get_tuple(xl, yl, xf, yf, phi):
    z1 = math.cos(np.deg2rad(phi)) * (xf - xl) + math.sin(np.deg2rad(phi)) * (yf - yl)
    z2 = -math.sin(np.deg2rad(phi)) * (xf - xl) + math.cos(np.deg2rad(phi)) * (yf - yl)
    return (round(z1), round(z2))

def obs_to_state_dqn(obs):
    Z1, Z2, Z3, Z4, Z5, Z6 = obs
    return np.array([
        Z1 / Q_TABLE_BASE_SIZE,
        Z2 / Q_TABLE_BASE_SIZE,
        Z3 / HEADING_MAX,
        Z4 / (ROLL_MAX - 1),
        Z5 / (ROLL_MAX - 1),
        Z6 / (CHOICES - 1)
    ], dtype=np.float32)

# -------------------------
# DQN architecture (must match training)
# -------------------------
class DQN(nn.Module):
    def __init__(self, in_dim=6, out_dim=CHOICES, hidden=128):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x):
        return self.net(x)

# -------------------------
# Prepare models & labels
# -------------------------
N_DQNS = max(1, len(DQN_MODEL_FILES))
labels = []
for i in range(N_DQNS):
    if i < len(DQN_LABELS) and DQN_LABELS[i]:
        labels.append(DQN_LABELS[i])
    else:
        labels.append(f"DQN #{i+1}")

# Build file list (if fewer files than labels, repeated last file will be used)
files = list(DQN_MODEL_FILES)
if len(files) < N_DQNS:
    if len(files) == 0:
        files = [None] * N_DQNS
    else:
        last = files[-1]
        files.extend([last] * (N_DQNS - len(files)))

device = torch.device("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None

dqn_models = []
model_loaded_flags = []
for i, fpath in enumerate(files):
    if TORCH_AVAILABLE and (fpath is not None) and os.path.exists(fpath):
        try:
            model = DQN().to(device)
            model.load_state_dict(torch.load(fpath, map_location=device))
            model.eval()
            dqn_models.append(model)
            model_loaded_flags.append(True)
            print(f"[{labels[i]}] loaded model from {fpath} on {device}")
        except Exception as e:
            print(f"[{labels[i]}] failed to load {fpath}: {e} — fallback to random")
            dqn_models.append(None)
            model_loaded_flags.append(False)
    else:
        if fpath is None:
            print(f"[{labels[i]}] no model path provided — using random policy")
        else:
            print(f"[{labels[i]}] model file not found ({fpath}) — using random policy")
        dqn_models.append(None)
        model_loaded_flags.append(False)

# -------------------------
# commands sequence (optional)
# -------------------------
com_seq = None
if COMMANDS_SEQUENCE_FILE and os.path.exists(COMMANDS_SEQUENCE_FILE):
    try:
        with open(COMMANDS_SEQUENCE_FILE, "rb") as f:
            com_seq = pickle.load(f)
        print("Loaded commands sequence from", COMMANDS_SEQUENCE_FILE)
    except Exception as e:
        print("Failed to load commands sequence:", e)

# -------------------------
# moving average helper
# -------------------------
def moving_average(x, w):
    """
    Returns same-length moving average using 'same' convolution.
    If w <= 1, returns original array.
    """
    if w is None or w <= 1:
        return np.array(x, dtype=float)
    # make sure window is an integer
    w = int(max(1, round(w)))
    kernel = np.ones(w, dtype=float) / w
    # use mode='same' to get same length; note edges are averaged with fewer points
    return np.convolve(x, kernel, mode='same')

# -------------------------
# Simulation run: collect per-DQN distances per step (averaged across episodes)
# -------------------------
print("Starting simulation: episodes =", HM_EPISODES, "steps/episode =", STEPS_PER_EPISODE)

# store per-DQN episode traces: list for each DQN -> list of episodes -> list of distances per step
dists_per_dqn = [ [] for _ in range(N_DQNS) ]
hits = [0] * N_DQNS

start_global = time.time()
for ep in range(HM_EPISODES):
    enemy = Plane(SIZE/2, SIZE/2)
    players = [Plane(enemy.x + SAFE_DISTANCE_HIGH, enemy.y + SAFE_DISTANCE_HIGH) for _ in range(N_DQNS)]

    # sync headings/rolls
    for p in players:
        p.heading = enemy.heading
        p.roll = enemy.roll

    episode_traces = [ [] for _ in range(N_DQNS) ]

    # run steps
    for t in range(STEPS_PER_EPISODE):
        # choose roll command from sequence if available, otherwise random (0..2)
        if com_seq is not None and t < len(com_seq):
            roll_command = com_seq[t]
        else:
            roll_command = np.random.randint(0, 3)

        # DQN agents decide & act
        for idx, p in enumerate(players):
            Z1_Z2 = get_tuple(enemy.x, enemy.y, p.x, p.y, enemy.heading * 360 / HEADING_MAX)
            Z1 = min(max(-Q_TABLE_BASE_SIZE, Z1_Z2[0]), Q_TABLE_BASE_SIZE - 1)
            Z2 = min(max(-Q_TABLE_BASE_SIZE, Z1_Z2[1]), Q_TABLE_BASE_SIZE - 1)
            Z3 = (enemy.heading - p.heading) % HEADING_MAX
            Z4 = enemy.get_discrete_roll()
            Z5 = p.get_discrete_roll()
            Z6 = roll_command
            obs = (Z1, Z2, Z3, Z4, Z5, Z6)

            if dqn_models[idx] is not None and TORCH_AVAILABLE:
                state = obs_to_state_dqn(obs)
                state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action = int(torch.argmax(dqn_models[idx](state_t)).item())
            else:
                action = np.random.randint(0, CHOICES)

            p.action(action)

        # leader moves after followers (same as training/previous scripts)
        enemy.action(roll_command)

        # record distances for each DQN this step
        for idx, p in enumerate(players):
            # convert to rounded grid coords for distance as in prior code
            p_x = int(round(p.x))
            p_y = int(round(p.y))
            e_x = int(round(enemy.x))
            e_y = int(round(enemy.y))
            distance = math.dist([p_x, p_y], [e_x, e_y])
            episode_traces[idx].append(distance)
            if p_x == e_x and p_y == e_y:
                hits[idx] += 1

    # append episode traces to global list
    for idx in range(N_DQNS):
        dists_per_dqn[idx].append(episode_traces[idx])

    if (ep + 1) % SHOW_EVERY == 0:
        elapsed = time.time() - start_global
        print(f"Episode {ep+1}/{HM_EPISODES} elapsed {elapsed:.1f}s")

# -------------------------
# Compute average distance per time-step per DQN (handle variable episode lengths)
# -------------------------
def avg_across_episodes(traces_list, steps=STEPS_PER_EPISODE):
    n_episodes = len(traces_list)
    avg = []
    for t in range(steps):
        s = 0.0
        count = 0
        for ep in traces_list:
            if t < len(ep):
                s += ep[t]
                count += 1
        avg.append(s / max(1, count))
    return avg

avg_traces = [ avg_across_episodes(dists_per_dqn[idx], steps=STEPS_PER_EPISODE) for idx in range(N_DQNS) ]

# compute smoothed traces
smoothed_traces = []
for idx in range(N_DQNS):
    sm = moving_average(avg_traces[idx], MOVING_AVERAGE_WINDOW)
    smoothed_traces.append(sm)

# -------------------------
# Optional: save per-model CSVs
# -------------------------
if SAVE_RESULTS_CSV:
    import pandas as pd
    for idx in range(N_DQNS):
        df = pd.DataFrame({"step": list(range(STEPS_PER_EPISODE)),
                           "distance_raw": avg_traces[idx],
                           "distance_smooth": smoothed_traces[idx]})
        filename = f"{OUT_CSV_PREFIX}_{idx+1}.csv"
        df.to_csv(filename, index=False)
        print("Saved", filename)

# -------------------------
# Plotting: raw (optional) + smoothed curves (bold)
# -------------------------
plt.figure(figsize=(12,8))
cmap = plt.get_cmap("tab10")
for idx in range(N_DQNS):
    clr = cmap(idx % 10)
    if PLOT_RAW:
        plt.plot(avg_traces[idx], color=clr, linewidth=1, alpha=0.25, label=None)  # faint raw
    plt.plot(smoothed_traces[idx], color=clr, linewidth=2.5, label=f"{labels[idx]} (loaded={model_loaded_flags[idx]})")

plt.hlines(y=SAFE_DISTANCE_LOW, xmin=0, xmax=STEPS_PER_EPISODE-1, colors='grey', linestyles='--', lw=1, label='b1')
plt.hlines(y=SAFE_DISTANCE_HIGH, xmin=0, xmax=STEPS_PER_EPISODE-1, colors='grey', linestyles='--', lw=1, label='b2')
plt.ylabel("Distance to leader (avg across episodes)")
plt.xlabel("Step")
plt.title(f"DQN comparison — {len(DQN_MODEL_FILES)} model files, averaged over {HM_EPISODES} episodes\n(moving average window={MOVING_AVERAGE_WINDOW})")
plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.0))
plt.tight_layout()
plt.show()

# -------------------------
# Final summary
# -------------------------
print("Done.")
for idx in range(N_DQNS):
    print(f"{labels[idx]}: episodes={len(dists_per_dqn[idx])}, total_hits={hits[idx]}, model_loaded={model_loaded_flags[idx]}")