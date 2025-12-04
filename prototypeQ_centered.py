import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
import math
import pandas as pd

style.use("ggplot")

#DONE: Mais resultados, YAW, distância média durante os episódios
#DONE: Trajetoria do lider pre-feita
#TODO: Aprendizado por reforço federado com Q-learning
#TODO: Diminuir a tabela

SIZE = 40
Q_TABLE_BASE_SIZE = 29 # 28 for 20 and 61 for 61
SHOW_SIZE = 400
HM_EPISODES = 2000000
MOVE_PENALTY = 1
ENEMY_PENALTY = 0 #10  # Bater no lider
CLOSE_ENEMY_PENALTY = 0# 5  # Degrau perto do lider
CLOSE_PENALTY_LINEAR = 100  # Slope perto do lider
FAR_PENALTY = 0 #5  # Degrau longe do lider
FAR_PENALTY_LINEAR = 100 # Slope longe do lider
SAFE_PLACE_REWARD = 0 # Recompensa de local seguro

iter_per_episode = 200

HEADING_MAX = 24
ROLL_MAX = 5
CHOICES = 3

epsilon = 0.9
EPS_DECAY = 0.9999
SHOW_EVERY = 1000
SHOW_ANY = False

SAFE_DISTANCE_LOW = 4 #6 #4
SAFE_DISTANCE_HIGH = 7 #9 #7

start_q_table = None #"qtable-planes-post-dir-fix2.pickle" # or filename

LEARNING_RATE = 0.4
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
        self.heading = np.random.randint(0, HEADING_MAX) # 0-7
        self.roll = 0

    def __str__(self):
        return f"{self.x}, {self.y}"

    def __sub__(self, other):
        return (self.x - other.x, self.y-other.y)

    def step(self, phi=False, follower=False):
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

        if(follower):
            follower.x -= self.speed * np.cos(np.deg2rad(self.heading*360/HEADING_MAX))
            follower.y -= self.speed * np.sin(np.deg2rad(self.heading*360/HEADING_MAX))
            
            if follower.x < 0:
                follower.x = 0
            elif follower.x > SIZE-1:
                follower.x = SIZE-1

            if follower.y < 0:
                follower.y = 0
            elif follower.y > SIZE-1:
                follower.y = SIZE-1



        else:
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

    
        

    def action(self, choice=None, follower=False):
        if choice == 0:
            self.step(-15, follower=follower)
        elif choice == 1:
            self.step(follower=follower)
        elif choice == 2:
            self.step(15, follower=follower)
        else:
            self.step(np.random.choice([-15, 0, 15]), follower=follower)
        
    
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



if start_q_table is None:
    q_table = {}
    q_table_speed = {}
    
    for x in range(-Q_TABLE_BASE_SIZE, Q_TABLE_BASE_SIZE):
        for y in range(-Q_TABLE_BASE_SIZE, Q_TABLE_BASE_SIZE):
            for hD in range(HEADING_MAX):
                for rL in range(ROLL_MAX):
                    for rF in range(ROLL_MAX):
                        for rC in range(ROLL_MAX):
                            Z = (x, y, hD, rL, rF, rC)
                            q_table[Z] = [np.random.uniform(-CHOICES, 0) for i in range(CHOICES)]
else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)


#print(q_table)
episode_rewards = []
episode_dif_rolls = []
episode_dif_headings = []
episode_distances = []

start_time = time.localtime()
start_time_text = time.strftime("%Y-%m-%d %H:%M:%S", start_time)
episode_time = []
episode_hits = []

print(f"Starting training for {HM_EPISODES} episodes at {start_time_text}")

for episode in range(HM_EPISODES):
    player = Plane()
    enemy = Plane(x=int(SIZE/2), y=int(SIZE/2))
    hit = 0
    #test = Plane(int(SIZE/2), int(SIZE/2))
    if episode % SHOW_EVERY == 0:
        print(f"on #{episode}, epsilon is {epsilon}")
        print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    episode_reward = 0
    episode_dif_roll = 0
    episode_dif_heading = 0
    episode_distance = 0
    for i in range(iter_per_episode):
        roll_command = np.random.randint(0, 3)

        player_aprox_X = int(np.round(player.x, decimals=0))
        player_aprox_Y = int(np.round(player.y, decimals=0))

        enemy_aprox_X = int(np.round(enemy.x, decimals=0))
        enemy_aprox_Y = int(np.round(enemy.y, decimals=0))

        Z1_Z2 = get_tuple(enemy.x, enemy.y, player.x, player.y, enemy.heading*360/HEADING_MAX)
        Z3 = (enemy.heading-player.heading)%HEADING_MAX
        Z4 = enemy.get_discrete_roll()
        Z5 = player.get_discrete_roll()
        Z6 = roll_command
        obs = (Z1_Z2[0], Z1_Z2[1], Z3, Z4, Z5, Z6)
        #obs = ((player_aprox_X-enemy_aprox_X, player_aprox_Y-enemy_aprox_Y), (player.heading-enemy.heading)%8, (player.get_discrete_roll()-enemy.get_discrete_roll()), (player.speed-enemy.speed+1))
        #print(obs)
        if np.random.random() > epsilon:
            # GET THE ACTION
            action = np.argmax(q_table[obs])
            #action_speed = np.argmax(q_table_speed[obs])
        else:
            if player.roll == 30:
                action = np.random.randint(0, CHOICES-1)
            elif player.roll == -30:
                action = np.random.randint(1, CHOICES)
            else:
                action = np.random.randint(0, CHOICES)

        # Take the action!
        player.action(action)
        #player.change_speed(action_speed)

        #### MAYBE ###
        enemy.action(roll_command, follower=player)
        ##############

        distance = math.dist([player_aprox_X, player_aprox_Y], [enemy_aprox_X, enemy_aprox_Y])

        # Calculate the reward

        beta = 0.5
        b1 = SAFE_DISTANCE_LOW
        b2 = SAFE_DISTANCE_HIGH
        p = math.sqrt(Z1_Z2[0]**2 + Z1_Z2[1]**2)
        d = max(b1-p, 0, p-b2)
        g = max(d, (b1*Z3)/(np.pi*(1+beta*d)))

        if player_aprox_X == enemy_aprox_X and player_aprox_Y == enemy_aprox_Y:
            reward = - (g + ENEMY_PENALTY)
            hit += 1
        elif SAFE_DISTANCE_LOW > distance:
            reward = -(g+CLOSE_ENEMY_PENALTY)
        elif SAFE_DISTANCE_LOW <= distance <= SAFE_DISTANCE_HIGH:
            reward = -(g+SAFE_PLACE_REWARD)
        else:
            reward = -(g+FAR_PENALTY)
            
        ## NOW WE KNOW THE REWARD, LET'S CALC YO
        # first we need to obs immediately after the move.  
        #           

        Z1_Z2 = get_tuple(enemy.x, enemy.y, player.x, player.y, enemy.heading*360/HEADING_MAX)
        Z3 = (enemy.heading-player.heading)%HEADING_MAX
        Z4 = enemy.get_discrete_roll()
        Z5 = player.get_discrete_roll()
        Z6 = roll_command
        new_obs = (Z1_Z2[0], Z1_Z2[1], Z3, Z4, Z5, Z6)
        #new_obs = ((player_aprox_X-enemy_aprox_X, player_aprox_Y-enemy_aprox_Y), (player.heading-enemy.heading)%8, (player.get_discrete_roll()-enemy.get_discrete_roll()), (player.speed-enemy.speed+1))

        max_future_q = np.max(q_table[new_obs])
        #max_future_q_speed = np.max(q_table_speed[new_obs])

        current_q = q_table[obs][action]
        #current_q_speed = q_table_speed[obs][action]

        if reward == SAFE_PLACE_REWARD:
            new_q = SAFE_PLACE_REWARD
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            #new_q_speed = (1 - LEARNING_RATE) * current_q_speed + LEARNING_RATE * (reward + DISCOUNT * max_future_q_speed)

        q_table[obs][action] = new_q
        #q_table_speed[obs][action] = new_q_speed

        show = show and SHOW_ANY  # show if we are in show mode and on time
        if show:
            player_aprox_X = int(np.round(player.x, decimals=0))
            player_aprox_Y = int(np.round(player.y, decimals=0))
            enemy_aprox_X = int(np.round(enemy.x, decimals=0))
            enemy_aprox_Y = int(np.round(enemy.y, decimals=0))

            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
            img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
            img = img.resize((SHOW_SIZE, SHOW_SIZE+100), resample=Image.NEAREST)  # resizing so we can see our agent in all its glory.
            
            ImageDraw.Draw(img).circle([enemy_aprox_Y*ZOOM,enemy_aprox_X*ZOOM], (SAFE_DISTANCE_HIGH+SAFE_DISTANCE_LOW)/2*ZOOM, fill=None, outline=None, width=SAFE_DISTANCE_HIGH*ZOOM-SAFE_DISTANCE_LOW*ZOOM)
            enemy.draw(color=colors[ENEMY_N])
            player.draw(color=colors[PLAYER_N])

            ImageDraw.Draw(img).rectangle([(0, SHOW_SIZE), (SHOW_SIZE, SHOW_SIZE+100)], fill=(255, 255, 255))  # draws a black rectangle around the image
            ImageDraw.Draw(img).text((10, SHOW_SIZE+10), f"Episode: {episode} - Step: {i}", fill=(0, 0, 0))
            ImageDraw.Draw(img).text((10, SHOW_SIZE+30), f"Reward: {reward} - Q: {new_q}", fill=(0, 0, 0))
            ImageDraw.Draw(img).text((10, SHOW_SIZE+50), f"Player - X: {player.x:.2f}, Y: {player.y:.2f}, H: {player.heading}, R: {player.roll} , S: {player.speed}", fill=(colors[PLAYER_N]))
            ImageDraw.Draw(img).text((10, SHOW_SIZE+70), f"Enemy - X: {enemy.x:.2f}, Y: {enemy.y:.2f}, H: {enemy.heading}, R: {enemy.roll} , S: {enemy.speed}", fill=(colors[ENEMY_N]))
            #ImageDraw.Draw(img).text((10, SHOW_SIZE+90), f"Test - X: {test.x:.2f}, Y: {test.y:.2f}, H: {test.heading}, HC: {test.heading*360/HEADING_MAX}, sin: {np.sin(np.deg2rad(test.heading*360/HEADING_MAX)):.2f}, cos: {np.cos(np.deg2rad(test.heading*360/HEADING_MAX)):.2f}, R: {test.roll} , S: {test.speed}", fill=(0, 0, 0))

            cv2.imshow("image", np.array(img))  # show it!
            #print(f"\n\nEpisode: {episode}, Reward: {reward}\nPlayer: X:{player_aprox_X}, Y:{player_aprox_Y}, Speed: {player.speed}\nEnemy: X:{enemy_aprox_X}, Y:{enemy_aprox_Y}\nDistance: {distance}")
            if reward == SAFE_PLACE_REWARD:  # crummy code to hang at the end if we reach abrupt end for good reasons or not.
                if cv2.waitKey(1000) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(1000) & 0xFF == ord('q'):
                    break

        episode_reward += reward
        episode_dif_roll += abs(player.get_discrete_roll() - enemy.get_discrete_roll())
        episode_dif_heading += Z3
        episode_distance += distance
        if player_aprox_X == enemy_aprox_X and player_aprox_Y == enemy_aprox_Y: # reward == -ENEMY_PENALTY
            break
            
    #print(f"Episode: {episode}, Reward: {reward}")

    #print(episode_reward)
    episode_rewards.append(episode_reward/iter_per_episode)
    episode_dif_rolls.append(episode_dif_roll/iter_per_episode)
    episode_dif_headings.append(episode_dif_heading/iter_per_episode)
    episode_distances.append(episode_distance/iter_per_episode)
    epsilon *= EPS_DECAY
    episode_time.append(time.mktime(time.localtime())-time.mktime(start_time))
    episode_hits.append(hit)

figure, axis = plt.subplots(2, 2)

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode="valid")
axis[0, 0].plot([i for i in range(len(moving_avg))], moving_avg, color="blue", label="Reward")
axis[0, 0].set_title("Reward")

moving_avg = np.convolve(episode_dif_rolls, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode="valid")
axis[0, 1].plot([i for i in range(len(moving_avg))], moving_avg, color="red", label="Roll")
axis[0, 1].set_title("Roll")


moving_avg = np.convolve(episode_dif_headings, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode="valid")
axis[1, 0].plot([i for i in range(len(moving_avg))], moving_avg, color="green", label="Heading")
axis[1, 0].set_title("Heading")

moving_avg = np.convolve(episode_distances, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode="valid")
axis[1, 1].plot([i for i in range(len(moving_avg))], moving_avg, color="orange", label="Distance")
axis[1, 1].set_title("Distance")

plt.show()

results = {'Reward': episode_rewards, 'Roll': episode_dif_rolls, 'Heading': episode_dif_headings, 'Distance': episode_distances, 'Time': episode_time, 'Hits': episode_hits}
df = pd.DataFrame(results)
df.to_csv(r'.\Tests\Q-Learning\qlearning_results_int.csv')


try:
    with open(f"qtable-planes-{int(time.time())}.pickle", "wb") as f:
        pickle.dump(q_table, f)
except Exception as e:
    with open(f"qtable-planes-redundant-{int(time.time())}.pickle", "wb") as f:
        pickle.dump(q_table, f)

print("Finished. Q-table saved.")