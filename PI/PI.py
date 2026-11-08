import numpy as np
import gym

def policy_evaluation(policy, env, V, gamma=0.99, theta=1e-8):

    num_states = env.observation_space.n
    P = env.unwrapped.P
    
    while True:
        delta = 0
        for s in range(num_states):
            v_old = V[s]
            
            a = policy[s]
            v_new = 0
            
            for prob, next_s, reward, done in P[s][a]:
                v_new += prob * (reward + gamma * V[next_s] * (1 - done))
                
            V[s] = v_new
            delta = max(delta, abs(v_old - V[s]))
            
        if delta < theta:
            break 
            
    return V

def policy_iteration(env, gamma=0.99, theta=1e-8):

    num_states = env.observation_space.n
    num_actions = env.action_space.n
    P = env.unwrapped.P
    
    V = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int) 
    
    iteration = 0
    while True:
        iteration += 1
        
        V = policy_evaluation(policy, env, V, gamma, theta)
        
        policy_stable = True
        
        for s in range(num_states):
            old_action = policy[s]
            
            q_values = np.zeros(num_actions)
            for a in range(num_actions):
                for prob, next_s, reward, done in P[s][a]:
                    q_values[a] += prob * (reward + gamma * V[next_s] * (1 - done))
            
            best_action = np.argmax(q_values)
            policy[s] = best_action
            
            if old_action != best_action:
                policy_stable = False
                
        if policy_stable:
            print(f"Policy Iteration đã hội tụ sau {iteration} bước (iterations)!")
            break
            
    return policy, V


if __name__ == "__main__":
    env = gym.make('FrozenLake-v1', is_slippery=True)
    
    print(f"Số trạng thái: {env.observation_space.n}")
    print(f"Số hành động: {env.action_space.n}")
    
    gamma_value = 0.99
    
    optimal_policy, optimal_V = policy_iteration(env, gamma=gamma_value)
    
    print("\nHàm giá trị tối ưu V*:")
    print(optimal_V.reshape(4, 4)) 
    
    print("\nChính sách tối ưu (0:Trái, 1:Xuống, 2:Phải, 3:Lên):")
    print(optimal_policy.reshape(4, 4))