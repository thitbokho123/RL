import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

policy = pd.read_csv('policy.csv', header=None).to_numpy()
gamma = 0.5
I = np.eye(policy.shape[0])
reward = pd.read_csv('reward.csv', header=None).to_numpy()
V = np.dot(np.linalg.inv(I - policy * gamma), reward) 
print(V)

