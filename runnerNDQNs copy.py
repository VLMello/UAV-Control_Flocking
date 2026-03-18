# runnerDrawTrials_multiDQN.py
# Adapted from your runnerDrawTrials.py to run N DQN agents in parallel and compare.
# Saves per-DQN traces and plots individual + mean DQN performance.

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
import math

style.use("ggplot")

# -------------------------
# Environment params
# -------------------------
SIZE = 400
Q_TABLE_BASE_SIZE = 29 # 28 for 20 and 61 for 61
M_CARLO_BASE_SIZE = 56 # 28 for 20 and 61 for 61
SHOW_SIZE = 800
HM_EPISODES = 2000           # reduce for quicker runs; increase for thorough stats
STEPS_PER_EPISODE = 300
MOVE_PENALTY = 1
ENEMY_PENALTY = 500
CLOSE_ENEMY_PENALTY = 0
CLOSE_PENALTY_LINEAR = 100
FAR_PENALTY = 0
FAR_PENALTY_LINEAR = 100
SAFE_PLACE_REWARD = 0

HEADING_MAX = 24
ROLL_MAX = 5
CHOICES = 3

epsilon = 0.0
EPS_DECAY = 0.99999
SHOW_EVERY = 100
SHOW_ROUNDED_POSITION = False

SAFE_DISTANCE_LOW = 5
SAFE_DISTANCE_HIGH = 9

start_q_table = ".\\Trained\\qtable-planes-1762740006.pickle"# "TreinamentoA.pickle" # or filename
start_mc_table = ".\\Trained\\mc-planes-1762831233.pickle" 
start_sarsa_table = ".\\Trained\\sarsa-planes-1762739879.pickle" 

commands_sequence = "commands-leader-1762474566.pickle"

LEARNING_RATE = 0.6
DISCOUNT = 0.8

Q_N = 1
SARSA_N = 2
ENEMY_N = 3
MC_N = 3

ZOOM = int(SHOW_SIZE/SIZE)

# -------------------------
# Multi-DQN parameters
# -------------------------
# Number of DQN agents to run in parallel
N_DQNS = 2

# Option A: list of model files for each DQN (provide exactly N_DQNS or fewer; missing entries will be handled)
DQN_MODEL_FILES = [
    "dqn_planes_1771871222.pth", # 0 - With leader command
    "dqn_planes_1771885804.pth"  # 1 - No leader command
    # Example: "dqn_planes_1768713482.pth",
    # Add your model paths here. If empty, script will try to duplicate DQN_MODEL_FILE below.
]

# Option B: single model file to duplicate N times (fallback when DQN_MODEL_FILES is empty)
DQN_MODEL_FILE = "dqn_planes_1768713482.pth"

# Color generation for many DQNs
cmap = plt.get_cmap("viridis")
dqn_colors = []
for i in range(N_DQNS):
    c = cmap(i / max(1, N_DQNS-1))
    # convert to 0-255 color tuple
    dqn_colors.append(tuple(int(255 * v) for v in c[:3]))

# base colors (keep earlier mapping)
colors = {
    1: (30, 30, 255),   # Q
    2: (30, 255, 30),   # SARSA
    3: (255, 255, 255), # enemy/background
    4: (255, 30, 30),   # MC
    # DQNs will use dqn_colors list for plotting / drawing
}

# -------------------------
# Torch & DQN model
# -------------------------
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

if not TORCH_AVAILABLE:
    print("Warning: PyTorch not available. DQN agents will fallback to random policy.")

# DQN network architecture (should match training)
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
# Environment classes / helpers
# -------------------------
class Plane:
    def __init__(self, x=None, y=None):
        if x is None:
            x = np.random.randint(0, SIZE)
        if y is None:
            y = np.random.randint(0, SIZE)
        self.x = x
        self.y = y
        self.speed = 2
        self.heading = 0
        self.roll = 0

    def __str__(self):
        return f"{self.x}, {self.y}"

    def __sub__(self, other):
        return (self.x - other.x, self.y-other.y)

    def step(self, phi=False):
        if phi:
            self.roll = max(min(self.roll + phi, 30), -30)

        if self.roll == 30:
            self.heading = (self.heading+2) % HEADING_MAX
        if self.roll == 15:
            self.heading = (self.heading+1) % HEADING_MAX
        elif self.roll == -15:
            self.heading = (self.heading-1) % HEADING_MAX
        elif self.roll == -30:
            self.heading = (self.heading-2) % HEADING_MAX

        self.x += self.speed * np.cos(np.deg2rad(self.heading*360/HEADING_MAX))
        self.y += self.speed * np.sin(np.deg2rad(self.heading*360/HEADING_MAX))

        if self.x < 0:
            self.x = 0
        elif self.x > SIZE-1:
            self.x = SIZE-1

        if self.y < 0:
            self.y = 0
        elif self.y > SIZE-1:
            self.y = SIZE-1

    def action(self, choice=None):
        if choice == 0:
            self.step(-15)
        elif choice == 1:
            self.step()
        elif choice == 2:
            self.step(15)
        else:
            self.step(np.random.choice([-15, 0, 15]))

    def change_speed(self, speed):
        if speed == 0:
            if self.speed > 1:
                self.speed -= 1
        elif speed == 2:
            if self.speed < 3:
                self.speed += 1

    def draw(self, color):
        forward_point = (self.y * ZOOM + 30 * np.sin(np.deg2rad(self.heading*360/HEADING_MAX-0)),
                         self.x * ZOOM + 30 * np.cos(np.deg2rad(self.heading*360/HEADING_MAX-0)))
        right_point = (self.y * ZOOM + 10 * np.sin(np.deg2rad(self.heading*360/HEADING_MAX+45)),
                       self.x * ZOOM + 10 * np.cos(np.deg2rad(self.heading*360/HEADING_MAX+45)))
        left_point = (self.y * ZOOM + 10 * np.sin(np.deg2rad(self.heading*360/HEADING_MAX-45)),
                      self.x * ZOOM + 10* np.cos(np.deg2rad(self.heading*360/HEADING_MAX-45)))
        ImageDraw.Draw(img).polygon([forward_point, right_point, (self.y*ZOOM,self.x*ZOOM), left_point], fill=color, outline=color)
        if SHOW_ROUNDED_POSITION:
            ImageDraw.Draw(img).circle([int(np.round(self.y, decimals=0)*ZOOM),int(np.round(self.x, decimals=0)*ZOOM)], 6, fill=color, outline=None, width=0)

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
            return -1

    def get_obs(self, other, roll_command):
        Z1_Z2 = get_tuple(other.x, other.y, self.x, self.y, other.heading*360/HEADING_MAX)
        Z1 = min(max(-Q_TABLE_BASE_SIZE, Z1_Z2[0]), Q_TABLE_BASE_SIZE-1)
        Z2 = min(max(-Q_TABLE_BASE_SIZE, Z1_Z2[1]), Q_TABLE_BASE_SIZE-1)
        Z3 = (other.heading-self.heading)%HEADING_MAX
        Z4 = other.get_discrete_roll()
        Z5 = self.get_discrete_roll()
        Z6 = roll_command
        obs = (Z1, Z2, Z3, Z4, Z5, Z6)
        return obs

def get_tuple(xl, yl, xf, yf, phi):
    z1 = math.cos(np.deg2rad(phi))*(xf-xl) + math.sin(np.deg2rad(phi))*(yf-yl)
    z2 = -math.sin(np.deg2rad(phi))*(xf-xl) + math.cos(np.deg2rad(phi))*(yf-yl)
    return (round(z1), round(z2))

# -------------------------
# Load tables & commands
# -------------------------
with open(start_q_table, "rb") as f:
    q_table = pickle.load(f)

with open(start_mc_table, "rb") as f:
    mc_table = pickle.load(f)

with open(start_sarsa_table, "rb") as f:
    sarsa_table = pickle.load(f)

with open(commands_sequence, "rb") as f:
    com_seq = pickle.load(f)

# -------------------------
# Load/prepare DQN models
# -------------------------
dqn_models = []
dqn_model_loaded_flags = []  # True if model present for each DQN
device = torch.device("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None

# Build list of files to attempt loading
files_to_try = []
if len(DQN_MODEL_FILES) > 0:
    files_to_try = DQN_MODEL_FILES[:N_DQNS]
# If fewer files provided than N_DQNS, and a single duplicate file is available, duplicate it
if len(files_to_try) < N_DQNS and os.path.exists(DQN_MODEL_FILE):
    needed = N_DQNS - len(files_to_try)
    files_to_try.extend([DQN_MODEL_FILE] * needed)

# Try to load each file; if not available or torch missing, append None (random fallback)
for idx in range(N_DQNS):
    model_path = files_to_try[idx] if idx < len(files_to_try) else None
    if TORCH_AVAILABLE and model_path is not None and os.path.exists(model_path):
        try:
            m = DQN().to(device)
            m.load_state_dict(torch.load(model_path, map_location=device))
            m.eval()
            dqn_models.append(m)
            dqn_model_loaded_flags.append(True)
            print(f"[DQN #{idx+1}] Loaded model: {model_path}")
        except Exception as e:
            print(f"[DQN #{idx+1}] Failed to load {model_path}: {e} -- falling back to random policy")
            dqn_models.append(None)
            dqn_model_loaded_flags.append(False)
    else:
        if model_path is None:
            print(f"[DQN #{idx+1}] No model path provided; using random policy.")
        else:
            print(f"[DQN #{idx+1}] Model file not found ({model_path}); using random policy.")
        dqn_models.append(None)
        dqn_model_loaded_flags.append(False)

# -------------------------
# Run episodes & collect traces (multi DQN supported)
# -------------------------
print("Starting episodes with N_DQNS =", N_DQNS)
distsQ = []
distsS = []
distsM = []
distsD_list = [ [] for _ in range(N_DQNS) ]  # per-DQN distances list of episodes

hitsQ = 0
hitsS = 0
hitsM = 0
hitsD = [0]*N_DQNS

episode_rewards = []

for episode in range(HM_EPISODES):
    enemy = Plane(np.floor(SIZE/2), np.floor(SIZE/2))
    player = Plane(enemy.x + SAFE_DISTANCE_HIGH,enemy.y + SAFE_DISTANCE_HIGH)   # Q
    player2 = Plane(enemy.x + SAFE_DISTANCE_HIGH,enemy.y + SAFE_DISTANCE_HIGH)  # MC
    player3 = Plane(enemy.x + SAFE_DISTANCE_HIGH,enemy.y + SAFE_DISTANCE_HIGH)  # SARSA
    players_dqn = [Plane(enemy.x + SAFE_DISTANCE_HIGH, enemy.y + SAFE_DISTANCE_HIGH) for _ in range(N_DQNS)]

    # sync initial heading/roll
    for p in (player, player2, player3):
        p.heading = enemy.heading
        p.roll = enemy.roll
    for p in players_dqn:
        p.heading = enemy.heading
        p.roll = enemy.roll

    env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    img = Image.fromarray(env, 'RGB')
    img = img.resize((SHOW_SIZE, SHOW_SIZE+120), resample=Image.NEAREST)

    if episode % SHOW_EVERY == 0:
        print(f'Episode {episode}/{HM_EPISODES}')
        show = True
    else:
        show = False

    episode_reward = 0
    enemy_dots = []
    player1_dots = []
    player2_dots = []
    player3_dots = []
    players_dqn_dots = [ [] for _ in range(N_DQNS) ]

    for i in range(STEPS_PER_EPISODE):
        # random-ish command as in original
        roll_command = np.random.randint(0, 7)
        if roll_command >= 2:
            roll_command = 1

        player_aprox_X = int(np.round(player.x, decimals=0))
        player_aprox_Y = int(np.round(player.y, decimals=0))

        enemy_aprox_X = int(np.round(enemy.x, decimals=0))
        enemy_aprox_Y = int(np.round(enemy.y, decimals=0))

        # Q-Learning follower (player)
        Z1_Z2 = get_tuple(enemy.x, enemy.y, player.x, player.y, enemy.heading*360/HEADING_MAX)
        Z1 = min(max(-Q_TABLE_BASE_SIZE, Z1_Z2[0]), Q_TABLE_BASE_SIZE-1)
        Z2 = min(max(-Q_TABLE_BASE_SIZE, Z1_Z2[1]), Q_TABLE_BASE_SIZE-1)
        Z3 = (enemy.heading-player.heading)%HEADING_MAX
        Z4 = enemy.get_discrete_roll()
        Z5 = player.get_discrete_roll()
        Z6 = roll_command
        obs = (Z1, Z2, Z3, Z4, Z5, Z6)
        action = np.argmax(q_table[obs])
        player.action(action)

        # MC follower (player2)
        Z1_Z2 = get_tuple(enemy.x, enemy.y, player2.x, player2.y, enemy.heading*360/HEADING_MAX)
        Z1 = min(max(-Q_TABLE_BASE_SIZE, Z1_Z2[0]), Q_TABLE_BASE_SIZE-1)
        Z2 = min(max(-Q_TABLE_BASE_SIZE, Z1_Z2[1]), Q_TABLE_BASE_SIZE-1)
        Z3 = (enemy.heading-player2.heading)%HEADING_MAX
        Z4 = enemy.get_discrete_roll()
        Z5 = player2.get_discrete_roll()
        Z6 = roll_command
        obs = (Z1, Z2, Z3, Z4, Z5, Z6)
        action = np.argmax(mc_table[obs])
        player2.action(action)

        # SARSA follower (player3)
        Z1_Z2 = get_tuple(enemy.x, enemy.y, player3.x, player3.y, enemy.heading*360/HEADING_MAX)
        Z1 = min(max(-Q_TABLE_BASE_SIZE, Z1_Z2[0]), Q_TABLE_BASE_SIZE-1)
        Z2 = min(max(-Q_TABLE_BASE_SIZE, Z1_Z2[1]), Q_TABLE_BASE_SIZE-1)
        Z3 = (enemy.heading-player3.heading)%HEADING_MAX
        Z4 = enemy.get_discrete_roll()
        Z5 = player3.get_discrete_roll()
        Z6 = roll_command
        obs = (Z1, Z2, Z3, Z4, Z5, Z6)
        action = np.argmax(sarsa_table[obs])
        player3.action(action)

        # Multi-DQN followers
        for idx in range(N_DQNS):
            p4 = players_dqn[idx]
            # build obs for this DQN follower
            Z1_Z2 = get_tuple(enemy.x, enemy.y, p4.x, p4.y, enemy.heading*360/HEADING_MAX)
            Z1 = min(max(-Q_TABLE_BASE_SIZE, Z1_Z2[0]), Q_TABLE_BASE_SIZE-1)
            Z2 = min(max(-Q_TABLE_BASE_SIZE, Z1_Z2[1]), Q_TABLE_BASE_SIZE-1)
            Z3 = (enemy.heading-p4.heading)%HEADING_MAX
            Z4 = enemy.get_discrete_roll()
            Z5 = p4.get_discrete_roll()
            Z6 = roll_command
            obs_dqn = (Z1, Z2, Z3, Z4, Z5, Z6)
            if dqn_models[idx] is not None and TORCH_AVAILABLE:
                state = obs_to_state_dqn(obs_dqn)
                state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action_dqn = int(torch.argmax(dqn_models[idx](state_t)).item())
                p4.action(action_dqn)
            else:
                # random fallback
                p4.action(np.random.randint(0, CHOICES))

        # leader moves
        enemy.action(roll_command)

        # compute distance & reward for primary (player) for metrics
        distance = math.dist([player_aprox_X, player_aprox_Y], [enemy_aprox_X, enemy_aprox_Y])

        beta = 0.5
        b1 = SAFE_DISTANCE_LOW
        b2 = SAFE_DISTANCE_HIGH
        p = math.sqrt(Z1_Z2[0]**2 + Z1_Z2[1]**2)
        d = max(b1-p, 0, p-b2)
        Z3_ = (enemy.heading-player.heading)%HEADING_MAX
        g = max(d, (b1*Z3_)/(np.pi*(1+beta*d)))
        reward = -g

        # record dots for averaging
        enemy_dots.append((enemy.x*ZOOM, enemy.y*ZOOM))
        player1_dots.append((player.x*ZOOM, player.y*ZOOM))
        player2_dots.append((player2.x*ZOOM, player2.y*ZOOM))
        player3_dots.append((player3.x*ZOOM, player3.y*ZOOM))
        for idx in range(N_DQNS):
            players_dqn_dots[idx].append((players_dqn[idx].x*ZOOM, players_dqn[idx].y*ZOOM))

        # hits
        if(int(enemy.x) == int(player.x) and int(enemy.y) == int(player.y)):
            hitsQ += 1
        if(int(enemy.x) == int(player2.x) and int(enemy.y) == int(player2.y)):
            hitsM += 1
        if(int(enemy.x) == int(player3.x) and int(enemy.y) == int(player3.y)):
            hitsS += 1
        for idx in range(N_DQNS):
            p4 = players_dqn[idx]
            if(int(enemy.x) == int(p4.x) and int(enemy.y) == int(p4.y)):
                hitsD[idx] += 1

        episode_reward += reward
        if reward == -ENEMY_PENALTY:
            break

    # compute distances time series (convert back to grid coords)
    enemy_dotsX = [dot[0]/ZOOM for dot in enemy_dots]
    enemy_dotsY = [SIZE - (dot[1]/ZOOM) for dot in enemy_dots]

    distances1 = []
    for i, dot in enumerate(player1_dots):
        distances1.append(math.dist([dot[0]/ZOOM, SIZE-(dot[1]/ZOOM)], [enemy_dotsX[i], enemy_dotsY[i]]))

    distances2 = []
    for i, dot in enumerate(player2_dots):
        distances2.append(math.dist([dot[0]/ZOOM, SIZE-(dot[1]/ZOOM)], [enemy_dotsX[i], enemy_dotsY[i]]))

    distances3 = []
    for i, dot in enumerate(player3_dots):
        distances3.append(math.dist([dot[0]/ZOOM, SIZE-(dot[1]/ZOOM)], [enemy_dotsX[i], enemy_dotsY[i]]))

    distances4_list = []
    for idx in range(N_DQNS):
        distances4 = []
        for i, dot in enumerate(players_dqn_dots[idx]):
            distances4.append(math.dist([dot[0]/ZOOM, SIZE-(dot[1]/ZOOM)], [enemy_dotsX[i], enemy_dotsY[i]]))
        distances4_list.append(distances4)

    distsQ.append(list(distances1))
    distsM.append(list(distances2))
    distsS.append(list(distances3))
    for idx in range(N_DQNS):
        distsD_list[idx].append(list(distances4_list[idx]))

    episode_rewards.append(episode_reward)

# -------------------------
# Compute average distance per time-step across episodes
# -------------------------
print("Episodes run:", HM_EPISODES)
print(f"Q-Learning hits: {hitsQ}")
print(f"Monte Carlo hits: {hitsM}")
print(f"SARSA hits: {hitsS}")
for idx in range(N_DQNS):
    print(f"DQN #{idx+1} hits: {hitsD[idx]} (model_loaded={dqn_model_loaded_flags[idx]})")

def avg_across_episodes(dlist, steps=STEPS_PER_EPISODE):
    # dlist: list of episodes, each a list of distances per step
    avg = []
    n = len(dlist)
    for i in range(steps):
        s = 0.0
        count = 0
        for ep in dlist:
            # some episodes may be shorter (if broken early); guard index
            if i < len(ep):
                s += ep[i]
                count += 1
        avg.append(s / max(1, count))
    return avg

distsQavg = avg_across_episodes(distsQ, steps=STEPS_PER_EPISODE)
distsMavg = avg_across_episodes(distsM, steps=STEPS_PER_EPISODE)
distsSavg = avg_across_episodes(distsS, steps=STEPS_PER_EPISODE)
distsDavg_list = [avg_across_episodes(distsD_list[idx], steps=STEPS_PER_EPISODE) for idx in range(N_DQNS)]

# group mean across DQNs
dqn_group_mean = []
for step in range(STEPS_PER_EPISODE):
    s = 0.0
    for idx in range(N_DQNS):
        s += distsDavg_list[idx][step]
    dqn_group_mean.append(s / N_DQNS)

# -------------------------
# Plot averages including all DQNs and mean
# -------------------------
plt.rcParams["figure.figsize"] = (12,8)
plt.plot(distsQavg, color="red", label="Q-Learning")
plt.plot(distsSavg, color="green", label="SARSA")
plt.plot(distsMavg, color="blue", label="Monte Carlo")

# individual DQNs (lighter colors)
for idx in range(N_DQNS):
    clr = tuple(c/255 for c in dqn_colors[idx])
    plt.plot(distsDavg_list[idx], color=clr, alpha=0.6, label=f"DQN #{idx+1}")

# DQN group mean (bold purple)
plt.plot(dqn_group_mean, color="purple", linewidth=3, label="DQN mean")

plt.hlines(y=SAFE_DISTANCE_LOW, xmin=0, xmax=STEPS_PER_EPISODE-1, colors='grey', linestyles='--', lw=2, label='b1')
plt.hlines(y=SAFE_DISTANCE_HIGH, xmin=0, xmax=STEPS_PER_EPISODE-1, colors='grey', linestyles='--', lw=2, label='b2')
plt.ylabel("Distância")
plt.xlabel("Passo")
plt.title(f"Comparação da distância entre o Líder e os Seguidores durante {HM_EPISODES} episódios (avg over episodes)")
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.18), fancybox=True, shadow=True, ncol=4)
plt.tight_layout()
plt.show()