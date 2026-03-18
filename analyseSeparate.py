# analyseSeparate.py (DQN-enabled)
# Updated to add DQN results: either load dqn_results CSV or simulate from saved model
# Original file: analyseSeparate.py. :contentReference[oaicite:0]{index=0}

import os
import time
import pickle
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# try to import torch (used only if we need to load the .pth and simulate)
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# --------------------------
# Paths to CSVs / model
# --------------------------
# Existing CSVs (you already have)
monte_csv = "Trained\\monte_carlo_results_int.csv"
q_csv = "Trained\\qlearning_results_int.csv"
sarsa_csv = "Trained\\sarsa_results_int.csv"

# DQN candidates (CSV first)
dqn_csv_candidates = [
    "dqn_results.csv",
    "dqn_results_1768851988.csv"
]

# If CSV not found, we'll try to load this trained model and simulate
dqn_model_file = "dqn_planes_1768713482.pth"  # as specified by user

# commands sequence used for deterministic leader commands (if available)
commands_candidates = [
    "commands-leader-1762474566.pickle",
    "commands-leader-1762468034.pickle",
    "commands-leader-1762468281.pickle",
    "commands-leader-1752508659.pickle"
]

# --------------------------
# Common plotting parameters
# --------------------------
MOVING_AVG_REWARD = 2000
MOVING_AVG_HITS = 3000
MOVING_AVG_HEADING = 2000
MOVING_AVG_DISTANCE = 2000
MOVING_AVG_TIME = 2000

# --------------------------
# Read baseline CSVs
# --------------------------
monte_carlo = pd.read_csv(monte_csv)
q_learning = pd.read_csv(q_csv)
sarsa = pd.read_csv(sarsa_csv)

# --------------------------
# Helper: attempt to load existing DQN CSV
# --------------------------
dqn_df = None
for path in dqn_csv_candidates:
    if os.path.exists(path):
        print(f"Found DQN CSV at: {path}")
        dqn_df = pd.read_csv(path)
        break

# --------------------------
# If no CSV, try to simulate using the .pth
# --------------------------
if dqn_df is None:
    print("No DQN CSV found. Will attempt to generate DQN metrics from the .pth model file.")
    if not TORCH_AVAILABLE:
        print("PyTorch is not available in this environment. Cannot load model to simulate.")
        print("Please install torch or provide dqn_results_int.csv in one of these locations:")
        for p in dqn_csv_candidates:
            print("  -", p)
        raise SystemExit("Missing torch and no DQN CSV found.")

    if not os.path.exists(dqn_model_file):
        print(f"Model file '{dqn_model_file}' not found in working directory.")
        print("Please place the trained model file there or create a CSV named dqn_results_int.csv.")
        raise SystemExit("DQN model file not found and no DQN CSV present.")

    # --------------------------
    # Recreate minimal environment from runner.py (consistent observation & rewards)
    # --------------------------
    SIZE = 80
    Q_TABLE_BASE_SIZE = 28
    SHOW_SIZE = 800
    HEADING_MAX = 24
    ROLL_MAX = 5
    CHOICES = 3

    SAFE_DISTANCE_LOW = 5
    SAFE_DISTANCE_HIGH = 9

    ENEMY_PENALTY = 500

    ZOOM = int(SHOW_SIZE / SIZE)

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
            # for distance calculation compatibility
            return type("V", (), {"x": self.x - other.x, "y": self.y - other.y})()

        def step(self, phi=False):
            if phi:
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

            if self.x < 0:
                self.x = 0
            elif self.x > SIZE - 1:
                self.x = SIZE - 1

            if self.y < 0:
                self.y = 0
            elif self.y > SIZE - 1:
                self.y = SIZE - 1

        def action(self, choice=None):
            if choice == 0:
                self.step(-15)
            elif choice == 1:
                self.step()
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
                return -1

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

    # --------------------------
    # DQN model definition (must match model architecture used for training)
    # --------------------------
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dqn_model = DQN().to(device)
    dqn_model.load_state_dict(torch.load(dqn_model_file, map_location=device))
    dqn_model.eval()

    # try to load a command sequence for deterministic leader behavior; fallback to random
    com_seq = None
    for cpath in commands_candidates:
        if os.path.exists(cpath):
            print("Found command sequence:", cpath)
            with open(cpath, "rb") as f:
                com_seq = pickle.load(f)
            break

    # number of episodes to simulate: use length of existing CSVs to match scale,
    # but limit to a reasonable max to avoid huge runs.
    target_len = min(len(q_learning['Reward']), len(monte_carlo['Reward']), len(sarsa['Reward']))
    MAX_EPISODES_SIM = 20000
    HM_EPISODES = min(target_len, MAX_EPISODES_SIM)
    if HM_EPISODES <= 0:
        HM_EPISODES = 1000

    print(f"Simulating {HM_EPISODES} episodes for DQN metrics. This may take a while.")

    # containers for simulated results
    rewards = []
    hits = []
    headings = []
    distances = []
    times = []

    start_time_global = time.time()

    for episode in range(HM_EPISODES):
        enemy = Plane(np.floor(SIZE / 2), np.floor(SIZE / 2))
        player = Plane(enemy.x + SAFE_DISTANCE_HIGH, enemy.y + SAFE_DISTANCE_HIGH)

        player.heading = enemy.heading
        player.roll = enemy.roll

        episode_reward = 0.0
        hit = 0
        t0 = time.time()

        steps = 120  # same as runner
        for i in range(steps):
            if com_seq is not None and i < len(com_seq):
                roll_command = com_seq[i]
            else:
                roll_command = np.random.randint(0, 3)

            # build observation for DQN follower
            Z1_Z2 = get_tuple(enemy.x, enemy.y, player.x, player.y, enemy.heading * 360 / HEADING_MAX)
            Z1 = min(max(-Q_TABLE_BASE_SIZE, Z1_Z2[0]), Q_TABLE_BASE_SIZE - 1)
            Z2 = min(max(-Q_TABLE_BASE_SIZE, Z1_Z2[1]), Q_TABLE_BASE_SIZE - 1)
            Z3 = (enemy.heading - player.heading) % HEADING_MAX
            Z4 = enemy.get_discrete_roll()
            Z5 = player.get_discrete_roll()
            Z6 = roll_command
            obs = (Z1, Z2, Z3, Z4, Z5, Z6)

            state = obs_to_state_dqn(obs)
            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = int(torch.argmax(dqn_model(state_t)).item())

            player.action(action)
            # leader moves
            enemy.action(roll_command)

            # compute reward per-step (same formula used in runner/ training)
            player_aprox_X = int(np.round(player.x, decimals=0))
            player_aprox_Y = int(np.round(player.y, decimals=0))

            enemy_aprox_X = int(np.round(enemy.x, decimals=0))
            enemy_aprox_Y = int(np.round(enemy.y, decimals=0))

            distance = math.dist([player_aprox_X, player_aprox_Y], [enemy_aprox_X, enemy_aprox_Y])

            beta = 0.5
            b1 = SAFE_DISTANCE_LOW
            b2 = SAFE_DISTANCE_HIGH
            p = math.sqrt(Z1_Z2[0] ** 2 + Z1_Z2[1] ** 2)
            d = max(b1 - p, 0, p - b2)
            Z3_ = (enemy.heading - player.heading) % HEADING_MAX
            g = max(d, (b1 * Z3_) / (np.pi * (1 + beta * d)))

            if player_aprox_X == enemy_aprox_X and player_aprox_Y == enemy_aprox_Y:
                reward = - (g + ENEMY_PENALTY)
                hit += 1
                # break? runner breaks out on hit; here we mark hit but continue steps to be consistent with recording
            elif SAFE_DISTANCE_LOW > distance:
                reward = - (g + 0)
            elif SAFE_DISTANCE_LOW <= distance <= SAFE_DISTANCE_HIGH:
                reward = - (g + 0)
            else:
                reward = - (g + 0)

            episode_reward += reward

        end_t = time.time()
        rewards.append(episode_reward)
        hits.append(hit)
        headings.append(Z3_)  # last-step heading diff as proxy
        distances.append(distance)
        times.append(end_t - t0)

        if (episode + 1) % 500 == 0:
            elapsed = time.time() - start_time_global
            print(f"Simulated {episode+1}/{HM_EPISODES} episodes (elapsed {elapsed:.1f}s)")

    # build dataframe matching existing CSV layout
    dqn_df = pd.DataFrame({
        "Reward": rewards,
        "Hits": hits,
        "Heading": headings,
        "Distance": distances,
        "Time": times
    })

    # Save CSV for later convenience
    out_csv = "dqn_results_int.csv"
    dqn_df.to_csv(out_csv, index=False)
    print(f"DQN simulated results saved to {out_csv}")

# --------------------------
# Now we have monte_carlo, q_learning, sarsa, and dqn_df
# Plot exactly like original script but include DQN
# --------------------------
figure, axis = plt.subplots(1, 1)

# Reward
moving_avg = np.convolve(monte_carlo['Reward'], np.ones((MOVING_AVG_REWARD,)) / MOVING_AVG_REWARD, mode="valid")
moving_avgM = moving_avg
axis.plot([i for i in range(len(moving_avg))], moving_avg, color="blue", label="Monte Carlo")

moving_avg = np.convolve(q_learning['Reward'], np.ones((MOVING_AVG_REWARD,)) / MOVING_AVG_REWARD, mode="valid")
moving_avgQ = moving_avg
axis.plot([i for i in range(len(moving_avg))], moving_avg, color="red", label="Q-Learning")

moving_avg = np.convolve(sarsa['Reward'], np.ones((MOVING_AVG_REWARD,)) / MOVING_AVG_REWARD, mode="valid")
moving_avgS = moving_avg
axis.plot([i for i in range(len(moving_avg))], moving_avg, color="green", label="SARSA")

moving_avg = np.convolve(dqn_df['Reward'], np.ones((MOVING_AVG_REWARD,)) / MOVING_AVG_REWARD, mode="valid")
moving_avgD = moving_avg
axis.plot([i for i in range(len(moving_avg))], moving_avg, color="purple", label="DQN")

axis.set_title("Reward")
plt.ylabel(f"Recompensa (média móvel {MOVING_AVG_REWARD} episódios)")
plt.xlabel("Epidódios")

axis.legend(["Monte Carlo", "Q-Learning", "SARSA", "DQN"], loc='upper right', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=4)

# Hits (binary collision rate)
def binary_hits(col):
    return [1 if v > 0 else 0 for v in col]

new_monte_carlo_hits = binary_hits(monte_carlo['Hits'])
new_q_learning_hits = binary_hits(q_learning['Hits'])
new_sarsa_hits = binary_hits(sarsa['Hits'])
new_dqn_hits = binary_hits(dqn_df['Hits'])

figure, axis = plt.subplots(1, 1)

moving_avg = np.convolve(new_monte_carlo_hits, np.ones((MOVING_AVG_HITS,)) / MOVING_AVG_HITS, mode="valid")
axis.plot([i for i in range(len(moving_avg))], moving_avg, color="blue", label="Monte Carlo")

moving_avg = np.convolve(new_q_learning_hits, np.ones((MOVING_AVG_HITS,)) / MOVING_AVG_HITS, mode="valid")
axis.plot([i for i in range(len(moving_avg))], moving_avg, color="red", label="Q-Learning")

moving_avg = np.convolve(new_sarsa_hits, np.ones((MOVING_AVG_HITS,)) / MOVING_AVG_HITS, mode="valid")
axis.plot([i for i in range(len(moving_avg))], moving_avg, color="green", label="SARSA")

moving_avg = np.convolve(new_dqn_hits, np.ones((MOVING_AVG_HITS,)) / MOVING_AVG_HITS, mode="valid")
axis.plot([i for i in range(len(moving_avg))], moving_avg, color="purple", label="DQN")

axis.set_title("Hits")
plt.ylabel(f"Taxa de Colisões (média móvel {MOVING_AVG_HITS} episódios)")
plt.xlabel("Epidódios")

axis.legend(["Monte Carlo", "Q-Learning", "SARSA", "DQN"], loc='upper right', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=4)

# Heading
figure, axis = plt.subplots(1, 1)

headingQ = [i * 15 for i in q_learning['Heading']]
headingS = [i * 15 for i in sarsa['Heading']]
headingM = [i * 15 for i in monte_carlo['Heading']]
headingD = [i * 15 for i in dqn_df['Heading']]

moving_avg = np.convolve(headingM, np.ones((MOVING_AVG_HEADING,)) / MOVING_AVG_HEADING, mode="valid")
axis.plot([i for i in range(len(moving_avg))], moving_avg, color="blue", label="Monte Carlo")

moving_avg = np.convolve(headingQ, np.ones((MOVING_AVG_HEADING,)) / MOVING_AVG_HEADING, mode="valid")
axis.plot([i for i in range(len(moving_avg))], moving_avg, color="red", label="Q-Learning")

moving_avg = np.convolve(headingS, np.ones((MOVING_AVG_HEADING,)) / MOVING_AVG_HEADING, mode="valid")
axis.plot([i for i in range(len(moving_avg))], moving_avg, color="green", label="SARSA")

moving_avg = np.convolve(headingD, np.ones((MOVING_AVG_HEADING,)) / MOVING_AVG_HEADING, mode="valid")
axis.plot([i for i in range(len(moving_avg))], moving_avg, color="purple", label="DQN")

axis.set_title("Heading")
plt.ylabel(f"Diferença de Heading (média móvel {MOVING_AVG_HEADING} episódios)")
plt.xlabel("Epidódios")

axis.legend(["Monte Carlo", "Q-Learning", "SARSA", "DQN"], loc='upper right', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=4)

# Distance
figure, axis = plt.subplots(1, 1)

moving_avg = np.convolve(monte_carlo['Distance'], np.ones((MOVING_AVG_DISTANCE,)) / MOVING_AVG_DISTANCE, mode="valid")
axis.plot([i for i in range(len(moving_avg))], moving_avg, color="blue", label="Monte Carlo")

moving_avg = np.convolve(q_learning['Distance'], np.ones((MOVING_AVG_DISTANCE,)) / MOVING_AVG_DISTANCE, mode="valid")
axis.plot([i for i in range(len(moving_avg))], moving_avg, color="red", label="Q-Learning")

moving_avg = np.convolve(sarsa['Distance'], np.ones((MOVING_AVG_DISTANCE,)) / MOVING_AVG_DISTANCE, mode="valid")
axis.plot([i for i in range(len(moving_avg))], moving_avg, color="green", label="SARSA")

moving_avg = np.convolve(dqn_df['Distance'], np.ones((MOVING_AVG_DISTANCE,)) / MOVING_AVG_DISTANCE, mode="valid")
axis.plot([i for i in range(len(moving_avg))], moving_avg, color="purple", label="DQN")

axis.set_title("Distance")
plt.ylabel(f"Distância (média móvel {MOVING_AVG_DISTANCE} episódios)")
plt.xlabel("Epidódios")

axis.legend(["Monte Carlo", "Q-Learning", "SARSA", "DQN"], loc='upper right', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=4)

# Time
figure, axis = plt.subplots(1, 1)

moving_avg = np.convolve(monte_carlo['Time'], np.ones((MOVING_AVG_TIME,)) / MOVING_AVG_TIME, mode="valid")
axis.plot([i for i in range(len(moving_avg))], moving_avg, color="blue", label="Monte Carlo")

moving_avg = np.convolve(q_learning['Time'], np.ones((MOVING_AVG_TIME,)) / MOVING_AVG_TIME, mode="valid")
axis.plot([i for i in range(len(moving_avg))], moving_avg, color="red", label="Q-Learning")

moving_avg = np.convolve(sarsa['Time'], np.ones((MOVING_AVG_TIME,)) / MOVING_AVG_TIME, mode="valid")
axis.plot([i for i in range(len(moving_avg))], moving_avg, color="green", label="SARSA")

moving_avg = np.convolve(dqn_df['Time'], np.ones((MOVING_AVG_TIME,)) / MOVING_AVG_TIME, mode="valid")
axis.plot([i for i in range(len(moving_avg))], moving_avg, color="purple", label="DQN")

axis.set_title("Time")
plt.ylabel(f"Tempo (média móvel {MOVING_AVG_TIME} episódios)")
plt.xlabel("Epidódios")

axis.legend(["Monte Carlo", "Q-Learning", "SARSA", "DQN"], loc='upper right', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=4)

# compute final stats and "90% episode" like original
mediaQfinal = sum(moving_avgQ[-10:-1]) / len(moving_avgQ[-10:-1])
mediaSfinal = sum(moving_avgS[-10:-1]) / len(moving_avgS[-10:-1])
mediaMfinal = sum(moving_avgM[-10:-1]) / len(moving_avgM[-10:-1])
mediaDfinal = sum(moving_avgD[-10:-1]) / len(moving_avgD[-10:-1])

targetQ = mediaQfinal - (mediaQfinal - moving_avgQ[0]) * 0.1
targetS = mediaSfinal - (mediaSfinal - moving_avgS[0]) * 0.1
targetM = mediaMfinal - (mediaMfinal - moving_avgM[0]) * 0.1
targetD = mediaDfinal - (mediaDfinal - moving_avgD[0]) * 0.1

noventaQ = 0
noventaS = 0
noventaM = 0
noventaD = 0

for i in range(len(moving_avgQ)):
    noventaQ = i
    if moving_avgQ[i] >= targetQ:
        break

for i in range(len(moving_avgS)):
    noventaS = i
    if moving_avgS[i] >= targetS:
        break

for i in range(len(moving_avgM)):
    noventaM = i
    if moving_avgM[i] >= targetM:
        break

for i in range(len(moving_avgD)):
    noventaD = i
    if moving_avgD[i] >= targetD:
        break

print("Q: Max Med:", mediaQfinal, "- Episode 90%:", noventaQ, "-Target 90:", targetQ, "- Time Final:", q_learning['Time'].iloc[-2], "- Time 90%:", q_learning['Time'].iloc[noventaQ])
print("S: Max Med:", mediaSfinal, "- Episode 90%:", noventaS, "-Target 90:", targetS, "- Time Final:", sarsa['Time'].iloc[-2], "- Time 90%:", sarsa['Time'].iloc[noventaS])
print("M: Max Med:", mediaMfinal, "- Episode 90%:", noventaM, "-Target 90:", targetM, "- Time Final:", monte_carlo['Time'].iloc[-2], "- Time 90%:", monte_carlo['Time'].iloc[noventaM])
print("D: Max Med:", mediaDfinal, "- Episode 90%:", noventaD, "-Target 90:", targetD, "- Time Final:", dqn_df['Time'].iloc[-2] if len(dqn_df['Time'])>1 else dqn_df['Time'].iloc[-1], "- Time 90%:", dqn_df['Time'].iloc[noventaD] if noventaD < len(dqn_df['Time']) else None)

plt.show()
