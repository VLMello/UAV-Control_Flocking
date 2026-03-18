"""
DQN version of prototypeQ_centered.py
Adapted from your original Q-learning script (prototypeQ_centered.py). :contentReference[oaicite:1]{index=1}
"""

# TODO ~ Dinâmica melhorada
#      ~ Ruído melhorado
#      ~ Retirada do Z6
#      - Testes de Deep Q-Network: Quantidade de neurônios e Camadas
#      - Double Deep Q-Learning
#      - Casos de falha (processo estocástico, cadeia de markov discreta)
#      - Vento


import random
import math
import time
from collections import deque, namedtuple
import numpy as np
from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt
import pickle
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# -------------------------
# Environment constants (kept from original)
# -------------------------
class DQN(nn.Module):
    def __init__(self, in_dim=6, out_dim=3, hidden=128):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class dqntrainer:

    def __init__(self):
        self.SIZE = 200
        self.SHOW_SIZE = 400
        self.HM_EPISODES = 10000      # smaller by default; increase if you want longer training
        self.iter_per_episode = 200

        self.HEADING_MAX = 90
        self.ROLL_MAX = 5
        self.CHOICES = 3

        self.epsilon = 0.9
        self.EPS_DECAY = 0.998        # decay per episode
        self.EPS_MIN = 0.02
        self.SHOW_EVERY = 500
        self.SHOW_ANY = False                                                           

        self.MAX_NOISE_STEP = 0.2 #0.2
        self.MAX_NOISE_HEADING = 0.4 #0.4
        self.MAX_NOISE_ROLL = 0.5 #1

        self.SAFE_DISTANCE_LOW = 4
        self.SAFE_DISTANCE_HIGH = 7

        self.MOVE_PENALTY = 1
        self.ENEMY_PENALTY = 0
        self.CLOSE_ENEMY_PENALTY = 0
        self.FAR_PENALTY = 0
        self.SAFE_PLACE_REWARD = 0

        self.Q_TABLE_BASE_SIZE = 29   # used for normalization of relative positions (matches original)

        # Reward shaping parameters (kept)
        self.beta = 0.5

        # -------------------------
        # PyTorch / DQN hyperparams
        # -------------------------
        self.GAMMA = 0.8
        self.LR = 1e-3
        self.BATCH_SIZE = 64
        self.MEMORY_SIZE = 20000
        self.TARGET_UPDATE_EVERY = 1000   # steps
        self.LEARN_EVERY = 1              # learn every N environment steps
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.ZOOM = int(self.SHOW_SIZE / self.SIZE)
        self.colors = {1: (255, 175, 0), 2: (0, 255, 0), 3: (0, 0, 255)}
        
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


    class Plane:
        def __init__(self, outer, x=None, y=None):
            if x is None:
                x = np.random.randint(0, outer.SIZE)
            if y is None:
                y = np.random.randint(0, outer.SIZE)
            self.x = x
            self.y = y
            self.speed = 1
            self.heading = np.random.randint(0, outer.HEADING_MAX)
            self.roll = 0

        def __str__(self):
            return f"{self.x}, {self.y}"

        def __sub__(self, other):
            return namedtuple("V", ["x", "y"])(self.x - other.x, self.y - other.y)

        def step(self, outer, phi=False, follower=False):
            noiseRoll = np.random.normal(-outer.MAX_NOISE_ROLL, outer.MAX_NOISE_ROLL)
            if phi is not False:
                self.roll = max(min(self.roll + phi + noiseRoll, 30), -30)

            sr = self.get_discrete_roll()
            noiseHead = np.random.normal(-outer.MAX_NOISE_HEADING, outer.MAX_NOISE_HEADING)
            if sr == 4:
                self.heading = (self.heading + 2 + noiseHead) % outer.HEADING_MAX
            if sr == 3:
                self.heading = (self.heading + 1 + noiseHead) % outer.HEADING_MAX 
            elif sr == 1:
                self.heading = (self.heading - 1 + noiseHead) % outer.HEADING_MAX
            elif sr == 0:
                self.heading = (self.heading - 2 + noiseHead) % outer.HEADING_MAX

            if follower:
                # follower parameter means "move follower opposite the leader motion" in original
                follower.x -= self.speed * np.cos(np.deg2rad(self.heading * 360 / outer.HEADING_MAX))
                follower.y -= self.speed * np.sin(np.deg2rad(self.heading * 360 / outer.HEADING_MAX))

                follower.x = float(max(0, min(outer.SIZE - 1, follower.x)))
                follower.y = float(max(0, min(outer.SIZE - 1, follower.y)))
            else:
                noiseStep = np.random.normal(-outer.MAX_NOISE_STEP, outer.MAX_NOISE_STEP)
                self.x += self.speed * np.cos(np.deg2rad(self.heading * 360 / outer.HEADING_MAX)) + noiseStep
                noiseStep = np.random.normal(-outer.MAX_NOISE_STEP, outer.MAX_NOISE_STEP)
                self.y += self.speed * np.sin(np.deg2rad(self.heading * 360 / outer.HEADING_MAX)) + noiseStep
                self.x = float(max(0, min(outer.SIZE - 1, self.x)))
                self.y = float(max(0, min(outer.SIZE - 1, self.y)))

        def action(self, outer1, choice=None, follower=False):
            if choice == 0:
                self.step(phi=-15, outer=outer1, follower=follower)
            elif choice == 1:
                self.step(follower=follower, outer=outer1)
            elif choice == 2:
                self.step(phi=15, outer=outer1, follower=follower)
            else:
                self.step(np.random.choice([-15, 0, 15]), follower=follower, outer=outer1)

        def get_discrete_roll(self):
            if self.roll < -22.5:
                return 0
            elif self.roll < -7.5:
                return 1
            elif self.roll < 7.5:
                return 2
            elif self.roll < 22.5:
                return 3
            elif self.roll >= 22.5:
                return 4
            else:
                return 2  # default

        def get_discrete_heading(self, outer):
            return int(self.heading % outer.HEADING_MAX)

    def get_tuple(self, xl, yl, xf, yf, phi):
        z1 = math.cos(np.deg2rad(phi)) * (xf - xl) + math.sin(np.deg2rad(phi)) * (yf - yl)
        z2 = -math.sin(np.deg2rad(phi)) * (xf - xl) + math.cos(np.deg2rad(phi)) * (yf - yl)
        return (round(z1), round(z2))

    # -------------------------
    # DQN network
    # -------------------------
    
    # -------------------------
    # Replay buffer
    # -------------------------

    class ReplayBuffer:
        def __init__(self, outer, capacity=20000):
            self.outer = outer
            self.memory = deque(maxlen=capacity)

        def push(self, *args):
            self.memory.append(self.outer.Transition(*args))

        def sample(self, batch_size):
            batch = random.sample(self.memory, batch_size)
            return self.outer.Transition(*zip(*batch))

        def __len__(self):
            return len(self.memory)

    # -------------------------
    # Helpers: state normalization
    # -------------------------
    def obs_to_state(self, obs):
        # obs = (Z1, Z2, Z3, Z4, Z5, Z6)
        z1, z2, z3, z4, z5, z6 = obs
        # normalize relative positions by self.Q_TABLE_BASE_SIZE (so inputs in approx [-1,1])
        s1 = float(z1) / float(self.Q_TABLE_BASE_SIZE)
        s2 = float(z2) / float(self.Q_TABLE_BASE_SIZE)
        s3 = float(z3) / float(self.HEADING_MAX)        # heading delta normalized
        s4 = float(z4) / float(self.ROLL_MAX - 1)       # discrete roll 0..4
        s5 = float(z5) / float(self.ROLL_MAX - 1)
        s6 = float(z6) / float(self.CHOICES - 1)        # roll_command 0..2
        return np.array([s1, s2, s3, s4, s5, s6], dtype=np.float32)

        

    # -------------------------
    # Training loop
    # -------------------------
    def train(self):
        # networks
        print(f"DQ - Using device: {self.DEVICE}")

        policy_net = DQN().to(self.DEVICE)
        target_net = DQN().to(self.DEVICE)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        optimizer = optim.Adam(policy_net.parameters(), lr=self.LR)
        memory = self.ReplayBuffer(outer=self)

        step_count = 0
        ep_rewards = []
        ep_dif_rolls = []
        ep_dif_headings = []
        ep_distances = []
        ep_times = []
        ep_hits = []
        start_time = time.time()

        eps = self.epsilon

        for episode in range(self.HM_EPISODES):
            player = self.Plane(outer=self)
            enemy = self.Plane(outer=self,x=int(self.SIZE/2), y=int(self.SIZE/2))
            hit = 0

            show = (episode % self.SHOW_EVERY == 0)
            episode_reward = 0.0
            episode_dif_roll = 0.0
            episode_dif_heading = 0.0
            episode_distance = 0.0

            for i in range(self.iter_per_episode):
                # random roll command for leader (same as original)
                roll_command = np.random.randint(0, 3)

                # compute current observation
                Z1_Z2 = self.get_tuple(enemy.x, enemy.y, player.x, player.y, enemy.get_discrete_heading(outer=self) * 360 / self.HEADING_MAX)
                Z3 = (enemy.get_discrete_heading(outer=self) - player.get_discrete_heading(outer=self)) % self.HEADING_MAX
                Z4 = enemy.get_discrete_roll()
                Z5 = player.get_discrete_roll()
                Z6 = roll_command
                obs = (Z1_Z2[0], Z1_Z2[1], Z3, Z4, Z5, Z6)

                state = self.obs_to_state(obs)
                state_t = torch.tensor(state, dtype=torch.float32, device=self.DEVICE).unsqueeze(0)

                # self.epsilon-greedy
                if random.random() > eps:
                    with torch.no_grad():
                        qvals = policy_net(state_t)
                        action = int(torch.argmax(qvals).item())
                else:
                    # respect roll saturation like original (prevent impossible actions)
                    if player.get_discrete_roll() == 4:
                        action = np.random.randint(0, self.CHOICES - 1)
                    elif player.get_discrete_roll() == 0:
                        action = np.random.randint(1, self.CHOICES)
                    else:
                        action = np.random.randint(0, self.CHOICES)

                # take action
                player.action(choice=action, outer1=self)
                enemy.action(choice=roll_command, outer1=self, follower=player)

                # compute reward
                player_aprox_X = int(round(player.x))
                player_aprox_Y = int(round(player.y))
                enemy_aprox_X = int(round(enemy.x))
                enemy_aprox_Y = int(round(enemy.y))
                distance = math.dist([player_aprox_X, player_aprox_Y], [enemy_aprox_X, enemy_aprox_Y])

                # reward shaping (copied)
                p = math.sqrt(Z1_Z2[0] ** 2 + Z1_Z2[1] ** 2)
                b1 = self.SAFE_DISTANCE_LOW
                b2 = self.SAFE_DISTANCE_HIGH
                d = max(b1 - p, 0, p - b2)
                g = max(d, (b1 * Z3) / (math.pi * (1 + self.beta * d)))

                if player_aprox_X == enemy_aprox_X and player_aprox_Y == enemy_aprox_Y:
                    reward = - (g + self.ENEMY_PENALTY)
                    hit += 1
                    done = True
                elif self.SAFE_DISTANCE_LOW > distance:
                    reward = - (g + self.CLOSE_ENEMY_PENALTY)
                    done = False
                elif self.SAFE_DISTANCE_LOW <= distance <= self.SAFE_DISTANCE_HIGH:
                    reward = - (g + self.SAFE_PLACE_REWARD)
                    done = False
                else:
                    reward = - (g + self.FAR_PENALTY)
                    done = False

                # next observation
                Z1_Z2_n = self.get_tuple(enemy.x, enemy.y, player.x, player.y, enemy.get_discrete_heading(outer=self) * 360 / self.HEADING_MAX)
                Z3_n = (enemy.get_discrete_heading(outer=self) - player.get_discrete_heading(outer=self)) % self.HEADING_MAX
                Z4_n = enemy.get_discrete_roll()
                Z5_n = player.get_discrete_roll()
                Z6_n = 0 #roll_command
                new_obs = (Z1_Z2_n[0], Z1_Z2_n[1], Z3_n, Z4_n, Z5_n, Z6_n)
                next_state = self.obs_to_state(new_obs)

                # store transition
                memory.push(state, action, reward, next_state, done)

                episode_reward += reward
                episode_dif_roll += abs(player.get_discrete_roll() - enemy.get_discrete_roll())
                episode_dif_heading += Z3
                episode_distance += distance

                # Learning step
                step_count += 1
                if len(memory) >= self.BATCH_SIZE and step_count % self.LEARN_EVERY == 0:
                    transitions = memory.sample(self.BATCH_SIZE)
                    batch = self.Transition(*transitions)

                    state_batch = torch.tensor(batch.state, dtype=torch.float32, device=self.DEVICE)
                    action_batch = torch.tensor(batch.action, dtype=torch.int64, device=self.DEVICE).unsqueeze(1)
                    reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=self.DEVICE).unsqueeze(1)
                    next_state_batch = torch.tensor(batch.next_state, dtype=torch.float32, device=self.DEVICE)
                    done_batch = torch.tensor(batch.done, dtype=torch.float32, device=self.DEVICE).unsqueeze(1)

                    # compute Q(s,a)
                    q_values = policy_net(state_batch).gather(1, action_batch)

                    # compute target: r + self.GAMMA * max_a' Q_target(next_state, a') * (1 - done)
                    with torch.no_grad():
                        next_q = target_net(next_state_batch).max(1)[0].unsqueeze(1)
                        target_q = reward_batch + (self.GAMMA * next_q * (1 - done_batch))

                    loss = F.mse_loss(q_values, target_q)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # update target network periodically (by steps)
                if step_count % self.TARGET_UPDATE_EVERY == 0:
                    target_net.load_state_dict(policy_net.state_dict())

                # optional rendering
                if show and self.SHOW_ANY:
                    env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)
                    img = Image.fromarray(env, 'RGB').resize((self.SHOW_SIZE, self.SHOW_SIZE + 100), resample=Image.NEAREST)
                    # draw safe ring
                    ImageDraw.Draw(img).ellipse(
                        [ (enemy_aprox_Y - (self.SAFE_DISTANCE_HIGH + self.SAFE_DISTANCE_LOW)/2) * self.ZOOM,
                        (enemy_aprox_X - (self.SAFE_DISTANCE_HIGH + self.SAFE_DISTANCE_LOW)/2) * self.ZOOM,
                        (enemy_aprox_Y + (self.SAFE_DISTANCE_HIGH + self.SAFE_DISTANCE_LOW)/2) * self.ZOOM,
                        (enemy_aprox_X + (self.SAFE_DISTANCE_HIGH + self.SAFE_DISTANCE_LOW)/2) * self.ZOOM],
                        outline=(150,150,150))
                    # draw enemy/player as in prototype (simple circle)
                    ImageDraw.Draw(img).ellipse([enemy_aprox_Y*self.ZOOM-5, enemy_aprox_X*self.ZOOM-5, enemy_aprox_Y*self.ZOOM+5, enemy_aprox_X*self.ZOOM+5], fill=self.colors[3])
                    ImageDraw.Draw(img).ellipse([player_aprox_Y*self.ZOOM-5, player_aprox_X*self.ZOOM-5, player_aprox_Y*self.ZOOM+5, player_aprox_X*self.ZOOM+5], fill=self.colors[1])
                    ImageDraw.Draw(img).rectangle([(0, self.SHOW_SIZE), (self.SHOW_SIZE, self.SHOW_SIZE+100)], fill=(255,255,255))
                    ImageDraw.Draw(img).text((10, self.SHOW_SIZE+10), f"Episode: {episode} - Step: {i}", fill=(0, 0, 0))
                    ImageDraw.Draw(img).text((10, self.SHOW_SIZE+30), f"Reward: {reward:.3f}", fill=(0,0,0))
                    ImageDraw.Draw(img).text((10, self.SHOW_SIZE+50), f"Player - X:{player.x:.2f}, Y:{player.y:.2f}", fill=(0,0,0))
                    ImageDraw.Draw(img).text((10, self.SHOW_SIZE+70), f"Eps: {eps:.3f}", fill=(0,0,0))
                    cv2.imshow("dqn", np.array(img))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        return policy_net, ep_rewards

                if done:
                    break

            # end episode
            ep_rewards.append(episode_reward / self.iter_per_episode)
            ep_dif_rolls.append(episode_dif_roll / self.iter_per_episode)
            ep_dif_headings.append(episode_dif_heading / self.iter_per_episode)
            ep_distances.append(episode_distance / self.iter_per_episode)
            ep_times.append(time.time() - start_time)
            ep_hits.append(hit)

            # decay self.epsilon
            eps = max(self.EPS_MIN, eps * self.EPS_DECAY)

            if episode % self.SHOW_EVERY == 0:
                print(f"DQ - Episode {episode}/{self.HM_EPISODES}, eps={eps:.4f}, recent mean reward {np.mean(ep_rewards[-min(50,len(ep_rewards)):])}")

        # After training: plot and save
        plt.figure(figsize=(10,8))
        ax1 = plt.subplot(2,2,1)
        ax1.plot(ep_rewards); ax1.set_title("Reward per episode")
        ax2 = plt.subplot(2,2,2)
        ax2.plot(ep_dif_rolls); ax2.set_title("Avg diff roll")
        ax3 = plt.subplot(2,2,3)
        ax3.plot(ep_dif_headings); ax3.set_title("Avg diff heading")
        ax4 = plt.subplot(2,2,4)
        ax4.plot(ep_distances); ax4.set_title("Avg distance")
        plt.tight_layout()
        plt.show()

        # save model and results
        torch.save(policy_net.state_dict(), f"dqn_planes_{int(time.time())}.pth")
        results = {'Reward': ep_rewards, 'Roll': ep_dif_rolls, 'Heading': ep_dif_headings, 'Distance': ep_distances, 'Time': ep_times, 'Hits': ep_hits}
        df = pd.DataFrame(results)
        df.to_csv('dqn_results.csv', index=False)

        print("Training finished. Model saved.")
        return policy_net, ep_rewards

    if __name__ == "__main__":
        #import math
        #print(f"Using device: {self.DEVICE}")
        policy_net, rewards = train()
