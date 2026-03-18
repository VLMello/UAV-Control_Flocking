import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
import math
import pandas as pd
import threading
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from prototypeQ_centered import Q_Learning
from prototype_MC_centered import monte_carlo
from prototypeSARSA_centered import sarsa
from prototype_DQN_noise import dqntrainer


if __name__ == "__main__":
    threads = []

    Q_trainer = Q_Learning()
    SARSA_trainer = sarsa()
    monte_trainer = monte_carlo()
    dqn_trainer = dqntrainer()

    t1 = threading.Thread(target=monte_trainer.train, args=(), kwargs={})
    threads.append(t1)

    t2 = threading.Thread(target=Q_trainer.train, args=(), kwargs={})
    threads.append(t2)

    t3 = threading.Thread(target=SARSA_trainer.train, args=(), kwargs={})
    threads.append(t3)

    t4 = threading.Thread(target=dqn_trainer.train, args=(), kwargs={})
    threads.append(t4)

    for t in threads:
        t.start()
    
    for t in threads:
        t.join()
