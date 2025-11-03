import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

monte_carlo = pd.read_csv("Tests\Monte Carlo\T7_NoStep\monte_carlo_results_int.csv")
q_learning = pd.read_csv("Tests\Q-Learning\T6_SemStep\qlearning_results_int.csv")
sarsa = pd.read_csv("Tests\SARSA\T5_NoStep\sarsa_results_int.csv")
#monte_carlo_long = pd.read_csv("Tests\Monte Carlo\TLarge1\monte_carlo_results_int.csv")


figure, axis = plt.subplots(2, 2)


#Reward
moving_avg = np.convolve(monte_carlo['Reward'], np.ones((2000,)) / 2000, mode="valid")
axis[0, 0].plot([i for i in range(len(moving_avg))], moving_avg, color="blue", label="Reward")

moving_avg = np.convolve(q_learning['Reward'], np.ones((2000,)) / 2000, mode="valid")
axis[0, 0].plot([i for i in range(len(moving_avg))], moving_avg, color="red", label="Reward")

moving_avg = np.convolve(sarsa['Reward'], np.ones((2000,)) / 2000, mode="valid")
axis[0, 0].plot([i for i in range(len(moving_avg))], moving_avg, color="green", label="Reward")
'''
moving_avg = np.convolve(monte_carlo_long['Reward'], np.ones((2000,)) / 2000, mode="valid")
axis[0, 0].plot([i for i in range(len(moving_avg))], moving_avg, color="orange", label="Reward")
'''
axis[0, 0].set_title("Reward")


#Roll
'''
moving_avg = np.convolve(monte_carlo['Roll'], np.ones((2000,)) / 2000, mode="valid")
axis[0, 1].plot([i for i in range(len(moving_avg))], moving_avg, color="blue", label="Roll")

moving_avg = np.convolve(q_learning['Roll'], np.ones((2000,)) / 2000, mode="valid")
axis[0, 1].plot([i for i in range(len(moving_avg))], moving_avg, color="red", label="Roll")

moving_avg = np.convolve(sarsa['Roll'], np.ones((2000,)) / 2000, mode="valid")
axis[0, 1].plot([i for i in range(len(moving_avg))], moving_avg, color="green", label="Roll")
axis[0, 1].set_title("Roll")
'''
#Hits
new_monte_carlo_hits = []
new_q_learning_hits = []
new_sarsa_hits = []
new_monte_carlo_long_hits = []

for i in range(len(monte_carlo['Hits'])):
    if monte_carlo['Hits'][i] > 0:
        new_monte_carlo_hits.append(1)
    else:
        new_monte_carlo_hits.append(0)

for i in range(len(q_learning['Hits'])):
    if q_learning['Hits'][i] > 0:
        new_q_learning_hits.append(1)
    else:
        new_q_learning_hits.append(0)

for i in range(len(sarsa['Hits'])):
    if sarsa['Hits'][i] > 0:
        new_sarsa_hits.append(1)
    else:
        new_sarsa_hits.append(0)
'''
for i in range(len(monte_carlo_long['Hits'])):
    if monte_carlo_long['Hits'][i] > 0:
        new_monte_carlo_long_hits.append(1)
    else:
        new_monte_carlo_long_hits.append(0)
'''
moving_avg = np.convolve(new_monte_carlo_hits, np.ones((3000,)) / 3000, mode="valid")
axis[0, 1].plot([i for i in range(len(moving_avg))], moving_avg, color="blue", label="Hits")

moving_avg = np.convolve(new_q_learning_hits, np.ones((3000,)) / 3000, mode="valid")
axis[0, 1].plot([i for i in range(len(moving_avg))], moving_avg, color="red", label="Hits")

moving_avg = np.convolve(new_sarsa_hits, np.ones((3000,)) / 3000, mode="valid")
axis[0, 1].plot([i for i in range(len(moving_avg))], moving_avg, color="green", label="Hits")
'''
moving_avg = np.convolve(new_monte_carlo_long_hits, np.ones((3000,)) / 3000, mode="valid")
axis[0, 1].plot([i for i in range(len(moving_avg))], moving_avg, color="orange", label="Hits")
'''
axis[0, 1].set_title("Hits")



#Heading
moving_avg = np.convolve(monte_carlo['Heading'], np.ones((2000,)) / 2000, mode="valid")
axis[1, 0].plot([i for i in range(len(moving_avg))], moving_avg, color="blue", label="Heading")

moving_avg = np.convolve(q_learning['Heading'], np.ones((2000,)) / 2000, mode="valid")
axis[1, 0].plot([i for i in range(len(moving_avg))], moving_avg, color="red", label="Heading")

moving_avg = np.convolve(sarsa['Heading'], np.ones((2000,)) / 2000, mode="valid")
axis[1, 0].plot([i for i in range(len(moving_avg))], moving_avg, color="green", label="Heading")
'''
moving_avg = np.convolve(monte_carlo_long['Heading'], np.ones((2000,)) / 2000, mode="valid")
axis[1, 0].plot([i for i in range(len(moving_avg))], moving_avg, color="orange", label="Heading")
'''
axis[1, 0].set_title("Heading")

#Distance
moving_avg = np.convolve(monte_carlo['Distance'], np.ones((2000,)) / 2000, mode="valid")
axis[1, 1].plot([i for i in range(len(moving_avg))], moving_avg, color="blue", label="Distance")

moving_avg = np.convolve(q_learning['Distance'], np.ones((2000,)) / 2000, mode="valid")
axis[1, 1].plot([i for i in range(len(moving_avg))], moving_avg, color="red", label="Distance")

moving_avg = np.convolve(sarsa['Distance'], np.ones((2000,)) / 2000, mode="valid")
axis[1, 1].plot([i for i in range(len(moving_avg))], moving_avg, color="green", label="Distance")
'''
moving_avg = np.convolve(monte_carlo_long['Distance'], np.ones((2000,)) / 2000, mode="valid")
axis[1, 1].plot([i for i in range(len(moving_avg))], moving_avg, color="orange", label="Distance")
'''
axis[1, 1].set_title("Distance")

axis[1, 0].legend(["Monte Carlo", "Q-Learning", "SARSA"], loc='upper right', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=3)

plt.show()