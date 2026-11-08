import numpy as np
import gym

def value_iteration(env, gamma=0.99, theta=1e-8):
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    P = env.unwrapped.P  
    
    V = np.zeros(num_states)
    
    while True:
        delta = 0 
        for s in range(num_states):
            v_old = V[s] 
            
            q_values = np.zeros(num_actions)
            for a in range(num_actions):
                for prob, next_s, reward, done in P[s][a]:
                    q_values[a] += prob * (reward + gamma * V[next_s] * (1 - done))
            
            V[s] = np.max(q_values)
            delta = max(delta, abs(v_old - V[s]))
            
        if delta < theta:
            print("Value Iteration đã hội tụ!")
            break
            
    return V

def extract_policy(V, env, gamma=0.99):
    env = env.unwrapped
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    
    policy = np.zeros(num_states, dtype=int)
    
    for s in range(num_states):
        q_values = np.zeros(num_actions)
        for a in range(num_actions):
            for prob, next_s, reward, done in env.P[s][a]:
                q_values[a] += prob * (reward + gamma * V[next_s] * (1 - done))
                
        policy[s] = np.argmax(q_values)
        
    return policy

env = gym.make('FrozenLake-v1', is_slippery=True)
gamma_value = 0.99

optimal_V = value_iteration(env, gamma=gamma_value)
optimal_policy = extract_policy(optimal_V, env, gamma=gamma_value)

print("\nHàm giá trị tối ưu V*:")
print(optimal_V.reshape(4, 4)) 

print("\nChính sách tối ưu (0:Trái, 1:Xuống, 2:Phải, 3:Lên):")
print(optimal_policy.reshape(4, 4))