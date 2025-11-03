import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
import math

style.use("ggplot")

SIZE = 80
Q_TABLE_BASE_SIZE = 28 # 28 for 20 and 61 for 61
M_CARLO_BASE_SIZE = 56 # 28 for 20 and 61 for 61
SHOW_SIZE = 800
HM_EPISODES = 100
MOVE_PENALTY = 1
ENEMY_PENALTY = 500#1000000
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
SHOW_EVERY = 1000
SHOW_ROUNDED_POSITION = True

SAFE_DISTANCE_LOW = 5
SAFE_DISTANCE_HIGH = 9

start_q_table = ".\Tests\Q-Learning\T4\qtable-planes-1759179536.pickle"# "TreinamentoA.pickle" # or filename
start_mc_table = ".\Tests\Monte Carlo\TLarge1\mc-planes-1759404375.pickle" 
start_sarsa_table = ".\Tests\SARSA\T3\sarsa-planes-1759177039.pickle" 

commands_sequence = "commands-leader-1752508659.pickle" # or filename

LEARNING_RATE = 0.6
DISCOUNT = 0.8

PLAYER_N = 1
FOOD_N = 2
ENEMY_N = 3

colors = {1: (255, 175, 0),
    2: (0, 255, 0),
    3: (0, 0, 255),
    4: (0, 255, 255)
}

ZOOM = int(SHOW_SIZE/SIZE)

class Plane:
    def __init__(self, x=None, y=None):
        if x is None:
            x = np.random.randint(0, SIZE)
        if y is None:
            y = np.random.randint(0, SIZE)
        self.x = x
        self.y = y
        self.speed = 2
        self.heading = 0 # 0-7
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

        # FIX: BOUNDARY CONDITIONS
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
        #print(f"Speed: {self.speed} - Action: {speed}")

    def draw(self, color):
        forward_point = (self.y * ZOOM + 30 * np.sin(np.deg2rad(self.heading*360/HEADING_MAX-0)), self.x * ZOOM + 30 * np.cos(np.deg2rad(self.heading*360/HEADING_MAX-0)))
        right_point = (self.y * ZOOM + 10 * np.sin(np.deg2rad(self.heading*360/HEADING_MAX+45)), self.x * ZOOM + 10 * np.cos(np.deg2rad(self.heading*360/HEADING_MAX+45)))
        left_point = (self.y * ZOOM + 10 * np.sin(np.deg2rad(self.heading*360/HEADING_MAX-45)), self.x * ZOOM + 10* np.cos(np.deg2rad(self.heading*360/HEADING_MAX-45)))
        #print(f"coords: {self} - heading: {self.heading} - roll: {self.roll}")
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


def dist(blobA, blobB):
    vector = blobA - blobB
    return math.sqrt(vector.x**2 + vector.y**2)

def get_tuple(xl, yl, xf, yf, phi):
    z1 = math.cos(np.deg2rad(phi))*(xf-xl) + math.sin(np.deg2rad(phi))*(yf-yl)
    z2 = -math.sin(np.deg2rad(phi))*(xf-xl) + math.cos(np.deg2rad(phi))*(yf-yl)
    return (round(z1), round(z2))

with open(start_q_table, "rb") as f:
    q_table = pickle.load(f)

with open(start_mc_table, "rb") as f:
    mc_table = pickle.load(f)

with open(start_sarsa_table, "rb") as f:
    sarsa_table = pickle.load(f)

with open(commands_sequence, "rb") as f:
    com_seq = pickle.load(f)

#print(q_table)
episode_rewards = []

for episode in range(HM_EPISODES):
    enemy = Plane(np.floor(SIZE/2), np.floor(SIZE/2))
    player = Plane(enemy.x + SAFE_DISTANCE_HIGH,enemy.y + SAFE_DISTANCE_HIGH)
    player2 = Plane(enemy.x + SAFE_DISTANCE_HIGH,enemy.y + SAFE_DISTANCE_HIGH)
    player3 = Plane(enemy.x + SAFE_DISTANCE_HIGH,enemy.y + SAFE_DISTANCE_HIGH)
    #player2 = Plane(enemy.x + SAFE_DISTANCE_HIGH,enemy.y - SAFE_DISTANCE_HIGH)

    player.heading = enemy.heading
    player.roll = enemy.roll

    player2.heading = enemy.heading
    player2.roll = enemy.roll

    player3.heading = enemy.heading
    player3.roll = enemy.roll

    env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
    img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
    img = img.resize((SHOW_SIZE, SHOW_SIZE+120), resample=Image.NEAREST)  # resizing so we can see our agent in all its glory.

    #test = Plane(int(SIZE/2), int(SIZE/2))
    if episode % SHOW_EVERY == 0:
        print(f"on #{episode}, epsilon is {epsilon}")
        print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    episode_reward = 0
    for i in range(20):
        roll_command = com_seq[i]

        player_aprox_X = int(np.round(player.x, decimals=0))
        player_aprox_Y = int(np.round(player.y, decimals=0))

        enemy_aprox_X = int(np.round(enemy.x, decimals=0))
        enemy_aprox_Y = int(np.round(enemy.y, decimals=0))

        Z1_Z2 = get_tuple(enemy.x, enemy.y, player.x, player.y, enemy.heading*360/HEADING_MAX)
        Z1 = min(max(-Q_TABLE_BASE_SIZE, Z1_Z2[0]), Q_TABLE_BASE_SIZE-1)
        Z2 = min(max(-Q_TABLE_BASE_SIZE, Z1_Z2[1]), Q_TABLE_BASE_SIZE-1)
        Z3 = (enemy.heading-player.heading)%HEADING_MAX
        Z4 = enemy.get_discrete_roll()
        Z5 = player.get_discrete_roll()
        Z6 = roll_command
        obs = (Z1, Z2, Z3, Z4, Z5, Z6)
        #obsFL = player.get_obs(enemy, roll_command)
        #obsFO = player.get_obs(player2, roll_command)
        

        #obs = ((player_aprox_X-enemy_aprox_X, player_aprox_Y-enemy_aprox_Y), (player.heading-enemy.heading)%8, (player.get_discrete_roll()-enemy.get_discrete_roll()), (player.speed-enemy.speed+1))
        #print(obs)
        # GET THE ACTION
        #action = np.argmax([np.argmax(q_table[obsFL],np.argmax(q_table[obsFO]))])
        action = np.argmax(q_table[obs])
        #print("Q obs:", obs, "result:", mc_table[obs], "action:", action)

        # Take the action!
        player.action(action)

        Z1_Z2 = get_tuple(enemy.x, enemy.y, player2.x, player2.y, enemy.heading*360/HEADING_MAX)
        Z1 = min(max(-M_CARLO_BASE_SIZE, Z1_Z2[0]), M_CARLO_BASE_SIZE-1)
        Z2 = min(max(-M_CARLO_BASE_SIZE, Z1_Z2[1]), M_CARLO_BASE_SIZE-1)
        Z3 = (enemy.heading-player2.heading)%HEADING_MAX
        Z4 = enemy.get_discrete_roll()
        Z5 = player2.get_discrete_roll()
        Z6 = roll_command
        obs = (Z1, Z2, Z3, Z4, Z5, Z6)
        #obsOL = player2.get_obs(enemy, roll_command)
        #obsOF = player2.get_obs(player, roll_command)

        # GET THE ACTION
        #action = np.argmax([np.argmax(q_table[obsOL],np.argmax(q_table[obsOF]))])
        action = np.argmax(mc_table[obs])
        #print("MC obs:", obs, "result:", mc_table[obs], "action:", action)
        player2.action(action)
        #player.change_speed(action_speed)

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

        #### MAYBE ###
        enemy.action(roll_command)
        ##############

        distance = math.dist([player_aprox_X, player_aprox_Y], [enemy_aprox_X, enemy_aprox_Y])

        beta = 0.5
        b1 = SAFE_DISTANCE_LOW
        b2 = SAFE_DISTANCE_HIGH
        p = math.sqrt(Z1_Z2[0]**2 + Z1_Z2[1]**2)
        d = max(b1-p, 0, p-b2)
        Z3 = (enemy.heading-player.heading)%HEADING_MAX
        g = max(d, (b1*Z3)/(np.pi*(1+beta*d)))
        reward = -g
        
        player_aprox_X = int(np.round(player.x, decimals=0))
        player_aprox_Y = int(np.round(player.y, decimals=0))
        player2_aprox_X = int(np.round(player2.x, decimals=0))
        player2_aprox_Y = int(np.round(player2.y, decimals=0))
        enemy_aprox_X = int(np.round(enemy.x, decimals=0))
        enemy_aprox_Y = int(np.round(enemy.y, decimals=0))

               
        enemy.draw(colors[ENEMY_N])
        player.draw(colors[PLAYER_N])
        player2.draw(colors[FOOD_N])
        player3.draw(colors[4])


        ImageDraw.Draw(img).rectangle([(0, SHOW_SIZE), (SHOW_SIZE, SHOW_SIZE+45)], fill=(255, 255, 255))  # draws a black rectangle around the image
        ImageDraw.Draw(img).text((10, SHOW_SIZE+10), f"Episode: {episode} - Step: {i}", fill=(0, 0, 0))
        ImageDraw.Draw(img).text((10, SHOW_SIZE+30), f"Reward: {reward} - Q: {q_table[obs]}", fill=(0, 0, 0))
        ImageDraw.Draw(img).text((10, SHOW_SIZE+50), f"Leader - X: {enemy.x:.2f}, Y: {enemy.y:.2f}, H: {enemy.heading}, R: {enemy.roll} , S: {enemy.speed}", fill=(colors[ENEMY_N]))
        ImageDraw.Draw(img).text((10, SHOW_SIZE+70), f"FollowerQL - X: {player.x:.2f}, Y: {player.y:.2f}, H: {player.heading}, R: {player.roll} , S: {player.speed}", fill=(colors[PLAYER_N]))
        ImageDraw.Draw(img).text((10, SHOW_SIZE+90), f"FollowerMC - X: {player2.x:.2f}, Y: {player2.y:.2f}, H: {player.heading}, R: {player.roll} , S: {player.speed}", fill=(colors[FOOD_N]))
        ImageDraw.Draw(img).text((10, SHOW_SIZE+110), f"FollowerSARSA - X: {player2.x:.2f}, Y: {player2.y:.2f}, H: {player.heading}, R: {player.roll} , S: {player.speed}", fill=(colors[4]))
        #ImageDraw.Draw(img).text((10, SHOW_SIZE+90), f"Test - X: {test.x:.2f}, Y: {test.y:.2f}, H: {test.heading}, HC: {test.heading*360/HEADING_MAX}, sin: {np.sin(np.deg2rad(test.heading*360/HEADING_MAX)):.2f}, cos: {np.cos(np.deg2rad(test.heading*360/HEADING_MAX)):.2f}, R: {test.roll} , S: {test.speed}", fill=(0, 0, 0))

        

        episode_reward += reward
        if reward == -ENEMY_PENALTY:
            break
    episode_rewards.append(episode_reward)
    cv2.imshow("image", np.array(img))  # show it!
    if cv2.waitKey(100) & 0xFF == ord('s'):
        SHOW_ROUNDED_POSITION = not SHOW_ROUNDED_POSITION
        cv2.waitKey(100)

    if reward == SAFE_PLACE_REWARD:  # crummy code to hang at the end if we reach abrupt end for good reasons or not.
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    else:
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        
moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode="valid")

plt.plot([i for i in range(len(moving_avg))], moving_avg, color="blue")
plt.ylabel(f"reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()