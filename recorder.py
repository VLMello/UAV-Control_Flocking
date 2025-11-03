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

SAFE_DISTANCE_LOW = 4
SAFE_DISTANCE_HIGH = 7

start_q_table = "qtable-planes-new-reward.pickle" # or filename

LEARNING_RATE = 0.6
DISCOUNT = 0.8

PLAYER_N = 1
FOOD_N = 2
ENEMY_N = 3

colors = {1: (255, 175, 0),
    2: (0, 255, 0),
    3: (0, 0, 255)
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
            ImageDraw.Draw(img).circle([int(np.round(self.y, decimals=0)*ZOOM),int(np.round(self.x, decimals=0)*ZOOM)], 5, fill=color, outline=None, width=0)
    
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


def dist(blobA, blobB):
    vector = blobA - blobB
    return math.sqrt(vector.x**2 + vector.y**2)

def get_tuple(xl, yl, xf, yf, phi):
    z1 = math.cos(np.deg2rad(phi))*(xf-xl) + math.sin(np.deg2rad(phi))*(yf-yl)
    z2 = -math.sin(np.deg2rad(phi))*(xf-xl) + math.cos(np.deg2rad(phi))*(yf-yl)
    return (round(z1), round(z2))



#print(q_table)

enemy = Plane(np.floor(SIZE/2), np.floor(SIZE/2))

commands = []

for i in range(200):
    #### MAYBE ###
    a = cv2.waitKey(200) & 0xFF
    if a == ord('a'):
        enemy.action(2)
        commands.append(2)
    elif a == ord('d'):
        enemy.action(0)
        commands.append(0)
    else:
        enemy.action(1)
        commands.append(1)
    ##############
    enemy_aprox_X = int(np.round(enemy.x, decimals=0))
    enemy_aprox_Y = int(np.round(enemy.y, decimals=0))

    env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
    img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
    img = img.resize((SHOW_SIZE, SHOW_SIZE+100), resample=Image.NEAREST)  # resizing so we can see our agent in all its glory.
    
    ImageDraw.Draw(img).circle([enemy_aprox_Y*ZOOM,enemy_aprox_X*ZOOM], (SAFE_DISTANCE_HIGH+SAFE_DISTANCE_LOW)/2*ZOOM, fill=None, outline=None, width=SAFE_DISTANCE_HIGH*ZOOM-SAFE_DISTANCE_LOW*ZOOM)
    enemy.draw(colors[ENEMY_N])

    ImageDraw.Draw(img).rectangle([(0, SHOW_SIZE), (SHOW_SIZE, SHOW_SIZE+100)], fill=(255, 255, 255))  # draws a black rectangle around the image
    ImageDraw.Draw(img).text((10, SHOW_SIZE+70), f"Enemy - X: {enemy.x:.2f}, Y: {enemy.y:.2f}, H: {enemy.heading}, R: {enemy.roll} , S: {enemy.speed}", fill=(colors[ENEMY_N]))

    cv2.imshow("image", np.array(img))  # show it!


with open(f"commands-leader-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(commands, f)

