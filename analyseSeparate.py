import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#monte_carlo = pd.read_csv("Tests\Monte Carlo\T13_v5_discount01_noStep\monte_carlo_results_int.csv")
#q_learning = pd.read_csv("Tests\Q-Learning\T12_Freeze_No_Step_size40\qlearning_results_int.csv")
#sarsa = pd.read_csv("Tests\SARSA\T12_Freeze_No_Step_size40\sarsa_results_int.csv")
#monte_carlo = pd.read_csv("Tests\Monte Carlo\T13_v3_discount03\monte_carlo_results_int.csv")
#q_learning = pd.read_csv("Tests\Q-Learning\T13_Freeze_Step_3_1_size40\qlearning_results_int.csv")
#sarsa = pd.read_csv("Tests\SARSA\T13_Freeze_Step_3_1_size40\sarsa_results_int.csv")

monte_carlo = pd.read_csv("Tests\Monte Carlo\T14_Step_3_1\monte_carlo_results_int.csv")
q_learning = pd.read_csv("Tests\Q-Learning\T14_Step_3_1\qlearning_results_int.csv")
sarsa = pd.read_csv("Tests\SARSA\T14_Step_3_1\sarsa_results_int.csv")
#q_learning = pd.read_csv("Tests\Q-Learning\T15_NoStep\qlearning_results_int.csv")
#sarsa = pd.read_csv("Tests\SARSA\T15_NoStep\sarsa_results_int.csv")
#monte_carlo = pd.read_csv("Tests\Monte Carlo\T15_NoStep\monte_carlo_results_int.csv")
#q_learning = pd.read_csv("Tests\Q-Learning\T16_fixedSkip_NoStep\qlearning_results_int.csv")
#sarsa = pd.read_csv("Tests\SARSA\T16_fixedSkip_NoStep\sarsa_results_int.csv")
#monte_carlo_long = pd.read_csv("Tests\Monte Carlo\TLarge1\monte_carlo_results_int.csv")


figure, axis = plt.subplots(1, 1)



#Reward
moving_avg = np.convolve(monte_carlo['Reward'], np.ones((2000,)) / 2000, mode="valid")
moving_avgM = moving_avg
axis.plot([i for i in range(len(moving_avg))], moving_avg, color="blue", label="Reward")

moving_avg = np.convolve(q_learning['Reward'], np.ones((2000,)) / 2000, mode="valid")
moving_avgQ = moving_avg
axis.plot([i for i in range(len(moving_avg))], moving_avg, color="red", label="Reward")

moving_avg = np.convolve(sarsa['Reward'], np.ones((2000,)) / 2000, mode="valid")
moving_avgS = moving_avg
axis.plot([i for i in range(len(moving_avg))], moving_avg, color="green", label="Reward")
'''
moving_avg = np.convolve(monte_carlo_long['Reward'], np.ones((2000,)) / 2000, mode="valid")
axis.plot([i for i in range(len(moving_avg))], moving_avg, color="orange", label="Reward")
'''
axis.set_title("Reward")
plt.ylabel(f"Recompensa (média móvel 2000 episódios)")
plt.xlabel("Epidódios")

axis.legend(["Monte Carlo", "Q-Learning", "SARSA"], loc='upper right', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=3)

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
figure, axis = plt.subplots(1, 1)

moving_avg = np.convolve(new_monte_carlo_hits, np.ones((3000,)) / 3000, mode="valid")
axis.plot([i for i in range(len(moving_avg))], moving_avg, color="blue", label="Hits")

moving_avg = np.convolve(new_q_learning_hits, np.ones((3000,)) / 3000, mode="valid")
axis.plot([i for i in range(len(moving_avg))], moving_avg, color="red", label="Hits")

moving_avg = np.convolve(new_sarsa_hits, np.ones((3000,)) / 3000, mode="valid")
axis.plot([i for i in range(len(moving_avg))], moving_avg, color="green", label="Hits")
'''
moving_avg = np.convolve(new_monte_carlo_long_hits, np.ones((3000,)) / 3000, mode="valid")
axis[0, 1].plot([i for i in range(len(moving_avg))], moving_avg, color="orange", label="Hits")
'''
axis.set_title("Hits")
plt.ylabel(f"Taxa de Colisões (média móvel 3000 episódios)")
plt.xlabel("Epidódios")

axis.legend(["Monte Carlo", "Q-Learning", "SARSA"], loc='upper right', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=3)




figure, axis = plt.subplots(1, 1)

#Heading
headingQ = [i*15 for i in q_learning['Heading']]
headingS = [i*15 for i in sarsa['Heading']]
headingM = [i*15 for i in monte_carlo['Heading']]
moving_avg = np.convolve(headingM, np.ones((2000,)) / 2000, mode="valid")
axis.plot([i for i in range(len(moving_avg))], moving_avg, color="blue", label="Heading")

moving_avg = np.convolve(headingQ, np.ones((2000,)) / 2000, mode="valid")
axis.plot([i for i in range(len(moving_avg))], moving_avg, color="red", label="Heading")

moving_avg = np.convolve(headingS, np.ones((2000,)) / 2000, mode="valid")
axis.plot([i for i in range(len(moving_avg))], moving_avg, color="green", label="Heading")
'''
moving_avg = np.convolve(monte_carlo_long['Heading'], np.ones((2000,)) / 2000, mode="valid")
axis[1, 0].plot([i for i in range(len(moving_avg))], moving_avg, color="orange", label="Heading")
'''
axis.set_title("Heading")
plt.ylabel(f"Diferença de Heading (média móvel 2000 episódios)")
plt.xlabel("Epidódios")

axis.legend(["Monte Carlo", "Q-Learning", "SARSA"], loc='upper right', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=3)


figure, axis = plt.subplots(1, 1)

#Distance
moving_avg = np.convolve(monte_carlo['Distance'], np.ones((2000,)) / 2000, mode="valid")
axis.plot([i for i in range(len(moving_avg))], moving_avg, color="blue", label="Distance")

moving_avg = np.convolve(q_learning['Distance'], np.ones((2000,)) / 2000, mode="valid")
axis.plot([i for i in range(len(moving_avg))], moving_avg, color="red", label="Distance")

moving_avg = np.convolve(sarsa['Distance'], np.ones((2000,)) / 2000, mode="valid")
axis.plot([i for i in range(len(moving_avg))], moving_avg, color="green", label="Distance")
'''
moving_avg = np.convolve(monte_carlo_long['Distance'], np.ones((2000,)) / 2000, mode="valid")
axis[1, 1].plot([i for i in range(len(moving_avg))], moving_avg, color="orange", label="Distance")
'''
axis.set_title("Distance")
plt.ylabel(f"Distância (média móvel 2000 episódios)")
plt.xlabel("Epidódios")

axis.legend(["Monte Carlo", "Q-Learning", "SARSA"], loc='upper right', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=3)


figure, axis = plt.subplots(1, 1)

#Time
moving_avg = np.convolve(monte_carlo['Time'], np.ones((2000,)) / 2000, mode="valid")
axis.plot([i for i in range(len(moving_avg))], moving_avg, color="blue", label="Time")

moving_avg = np.convolve(q_learning['Time'], np.ones((2000,)) / 2000, mode="valid")
axis.plot([i for i in range(len(moving_avg))], moving_avg, color="red", label="Time")

moving_avg = np.convolve(sarsa['Time'], np.ones((2000,)) / 2000, mode="valid")
axis.plot([i for i in range(len(moving_avg))], moving_avg, color="green", label="Time")
'''
moving_avg = np.convolve(monte_carlo_long['Distance'], np.ones((2000,)) / 2000, mode="valid")
axis[1, 1].plot([i for i in range(len(moving_avg))], moving_avg, color="orange", label="Distance")
'''
axis.set_title("Time")
plt.ylabel(f"Tempo (média móvel 2000 episódios)")
plt.xlabel("Epidódios")

axis.legend(["Monte Carlo", "Q-Learning", "SARSA"], loc='upper right', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=3)



mediaQfinal = sum(moving_avgQ[-10:-1])/len(moving_avgQ[-10:-1])
mediaSfinal = sum(moving_avgS[-10:-1])/len(moving_avgS[-10:-1])
mediaMfinal = sum(moving_avgM[-10:-1])/len(moving_avgM[-10:-1])

targetQ = mediaQfinal-(mediaQfinal-moving_avgQ[0])*0.1
targetS = mediaSfinal-(mediaSfinal-moving_avgS[0])*0.1
targetM = mediaMfinal-(mediaMfinal-moving_avgM[0])*0.1

noventaQ = 0
noventaS = 0
noventaM = 0

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

print("Q: Max Med:", mediaQfinal, "- Episode 90%:", noventaQ, "-Target 90:", targetQ, "- Time Final:", q_learning['Time'][len(q_learning['Time'])-2], "- Time 90%:", q_learning['Time'][noventaQ])
print("S: Max Med:", mediaSfinal, "- Episode 90%:", noventaS, "-Target 90:", targetS, "- Time Final:", sarsa['Time'][len(sarsa['Time'])-2], "- Time 90%:", sarsa['Time'][noventaS])
print("M: Max Med:", mediaMfinal, "- Episode 90%:", noventaM, "-Target 90:", targetM, "- Time Final:", monte_carlo['Time'][len(monte_carlo['Time'])-2], "- Time 90%:", monte_carlo['Time'][noventaM])

plt.show()