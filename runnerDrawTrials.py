import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
import math

style.use("ggplot")

SIZE = 40
Q_TABLE_BASE_SIZE = 29 # 28 for 20 and 61 for 61
M_CARLO_BASE_SIZE = 56 # 28 for 20 and 61 for 61
SHOW_SIZE = 800
HM_EPISODES = 10000
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
SHOW_EVERY = 1
SHOW_ROUNDED_POSITION = False

SAFE_DISTANCE_LOW = 5
SAFE_DISTANCE_HIGH = 9

start_q_table = ".\Tests\Q-Learning\T12_Freeze_No_Step_size40\qtable-planes-1762454062.pickle"# "TreinamentoA.pickle" # or filename
start_mc_table = ".\Tests\Monte Carlo\T13_v6_discount03_noStep\mc-planes-1762882272.pickle" 
start_sarsa_table = ".\Tests\SARSA\T12_Freeze_No_Step_size40\sarsa-planes-1762454048.pickle" 

##start_q_table = ".\Tests\Q-Learning\T12_Freeze_No_Step_size40\qtable-planes-1762454062.pickle"# "TreinamentoA.pickle" # or filename
##start_mc_table = ".\Tests\Monte Carlo\T12_Freeze_No_Step_size40\mc-planes-1762519034.pickle" 
##start_sarsa_table = ".\Tests\SARSA\T12_Freeze_No_Step_size40\sarsa-planes-1762454048.pickle" 

#start_q_table = ".\Tests\Q-Learning\T13_Freeze_Step_3_1_size40\qtable-planes-1762740006.pickle"# "TreinamentoA.pickle" # or filename
##start_mc_table = ".\Tests\Monte Carlo\T13_Freeze_Step_3_1_size40\mc-planes-1762739606.pickle" 
##start_mc_table = ".\Tests\Monte Carlo\T13_v2_discount06\mc-planes-1762776293.pickle" 
##start_mc_table = ".\Tests\Monte Carlo\T13_v3_discount03\mc-planes-1762800416.pickle" 
#start_mc_table = ".\Tests\Monte Carlo\T13_v4_discount01\mc-planes-1762831233.pickle" 
#start_sarsa_table = ".\Tests\SARSA\T13_Freeze_Step_3_1_size40\sarsa-planes-1762739879.pickle" 

#commands_sequence = "commands-leader-1752508659.pickle" # or filename
#commands_sequence = "commands-leader-1762468034.pickle" # or filename
#commands_sequence = "commands-leader-1762468281.pickle" # or filename
commands_sequence = "commands-leader-1762474566.pickle" # or filename

LEARNING_RATE = 0.6
DISCOUNT = 0.8

Q_N = 1
SARSA_N = 2
ENEMY_N = 3
MC_N = 3

colors = {1: (30, 30, 255),
    2: (30, 255, 30),
    3: (255, 255, 255),
    4: (255, 30, 30)
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

distsQ = []
distsS = []
distsM = []

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
        #print(f"on #{episode}, epsilon is {epsilon}")
        #print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    episode_reward = 0
    enemy_dots = []
    player1_dots = []
    player2_dots = []
    player3_dots = []

    for i in range(300):
        roll_command = np.random.randint(0, 7)
        if roll_command >= 2:
            roll_command = 1

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
        Z1 = min(max(-Q_TABLE_BASE_SIZE, Z1_Z2[0]), Q_TABLE_BASE_SIZE-1)
        Z2 = min(max(-Q_TABLE_BASE_SIZE, Z1_Z2[1]), Q_TABLE_BASE_SIZE-1)
        Z3 = (enemy.heading-player2.heading)%HEADING_MAX
        Z4 = enemy.get_discrete_roll()
        Z5 = player2.get_discrete_roll()
        Z6 = roll_command
        obs = (Z1, Z2, Z3, Z4, Z5, Z6)

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

               
     
        enemy_dots.append((enemy.x*ZOOM, enemy.y*ZOOM))
        player1_dots.append((player.x*ZOOM, player.y*ZOOM))
        player2_dots.append((player2.x*ZOOM, player2.y*ZOOM))
        player3_dots.append((player3.x*ZOOM, player3.y*ZOOM))

        episode_reward += reward
        if reward == -ENEMY_PENALTY:
            break

    enemy_dotsX = []
    enemy_dotsY = []
    for dot in enemy_dots:
        enemy_dotsX.append(dot[0]/ZOOM)
        enemy_dotsY.append(SIZE-(dot[1]/ZOOM))

    distances1 = []
    i=0
    for dot in player1_dots:
        distances1.append(math.dist([dot[0]/ZOOM, SIZE-(dot[1]/ZOOM)], [enemy_dotsX[i], enemy_dotsY[i]]))
        i+=1

    distances2 = []
    i=0
    for dot in player2_dots:
        distances2.append(math.dist([dot[0]/ZOOM, SIZE-(dot[1]/ZOOM)], [enemy_dotsX[i], enemy_dotsY[i]]))
        i+=1

    distances3 = []
    i=0
    for dot in player3_dots:
        distances3.append(math.dist([dot[0]/ZOOM, SIZE-(dot[1]/ZOOM)], [enemy_dotsX[i], enemy_dotsY[i]]))
        i+=1

    distsQ.append(list(distances1))
    distsS.append(list(distances3))
    distsM.append(list(distances2))

    

    episode_rewards.append(episode_reward)

#print(distsQ)


print("Teorico: ", str(len(distsQ)), "por", HM_EPISODES)
print("QL: ", str(len(distsQ)), "por", str(len(distsQ[0])))
distsQavg = []
for i in range(300):
    totalQ = 0
    for j in range(HM_EPISODES):
        totalQ += distsQ[j][i]
    distsQavg.append(totalQ/len(distsQ))

distsSavg = []
for i in range(300):
    totalS = 0
    for j in range(HM_EPISODES):
        totalS += distsS[j][i]
    distsSavg.append(totalS/len(distsS))

distsMavg = []
for i in range(300):
    totalM = 0
    for j in range(HM_EPISODES):
        totalM += distsM[j][i]
    distsMavg.append(totalM/len(distsM))



plt.rcParams["figure.figsize"] = (10,10)
plt.plot(distsQavg, color="red", label="Q-Learning")
plt.plot(distsSavg, color="green", label="SARSA")
plt.plot(distsMavg, color="blue", label="Monte Carlo")
plt.hlines(y=4, xmin=0, xmax=299, colors='grey', linestyles='--', lw=2, label='b1')
plt.hlines(y=7, xmin=0, xmax=299, colors='grey', linestyles='--', lw=2, label='b2')
plt.ylabel(f"Distância")
plt.xlabel("Passo")
plt.title("Comparação da distância entre o Líder e os Seguidores durante " + str(HM_EPISODES) + " episódios")
plt.legend(["Q-Learning", "SARSA", "Monte Carlo"], loc='lower center', bbox_to_anchor=(0.5, -0.11),
          fancybox=True, shadow=True, ncol=3)
plt.show()