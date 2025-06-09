import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

def train_q_learning(env, episodes, alpha, gamma, max_epsilon, min_epsilon, decay_rate):
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    rewards = []
    epsilon = max_epsilon

    print("Rozpoczynanie treningu Q-learning...")
    for episode in range(episodes):
        state, info = env.reset()
        done = False
        total_reward = 0

        while not done:
           
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  
            else:
                action = np.argmax(q_table[state]) 
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            
            new_value = old_value + alpha * (reward + gamma * next_max - old_value)
            q_table[state, action] = new_value

            state = next_state
            total_reward += reward
        
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        rewards.append(total_reward)

        if (episode + 1) % (episodes // 10) == 0:
            print(f"Q-learning: Ukończono epizod {episode + 1}/{episodes}")
            
    return q_table, rewards

def train_sarsa(env, episodes, alpha, gamma, max_epsilon, min_epsilon, decay_rate):
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    rewards = []
    epsilon = max_epsilon

    print("\nRozpoczynanie treningu SARSA...")
    for episode in range(episodes):
        state, info = env.reset()
        done = False
        total_reward = 0

        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        while not done:
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if np.random.uniform(0, 1) < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(q_table[next_state])
            
            old_value = q_table[state, action]
            next_q_value = q_table[next_state, next_action]
            
            new_value = old_value + alpha * (reward + gamma * next_q_value - old_value)
            q_table[state, action] = new_value

            state = next_state
            action = next_action  
            total_reward += reward
        
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        rewards.append(total_reward)
        
        if (episode + 1) % (episodes // 10) == 0:
            print(f"SARSA: Ukończono epizod {episode + 1}/{episodes}")

    return q_table, rewards

def visualize_agent(env_name, q_table):
    print(f"\nUruchamianie wizualizacji dla agenta w środowisku {env_name}...")
    env_vis = gym.make(env_name, render_mode="human")
    state, info = env_vis.reset()
    done = False
    
    while not done:
        action = np.argmax(q_table[state]) 
        state, reward, terminated, truncated, info = env_vis.step(action)
        done = terminated or truncated
        time.sleep(0.1) 
    
    env_vis.close()
    print("Wizualizacja zakończona.")


if __name__ == "__main__":
   
    env_name = "Taxi-v3"
    env = gym.make(env_name)

    episodes = 20000  
    alpha = 0.1       
    gamma = 0.9      
    
    max_epsilon = 1.0
    min_epsilon = 0.01
    decay_rate = 0.0005 

    q_table_q_learning, rewards_q_learning = train_q_learning(env, episodes, alpha, gamma, max_epsilon, min_epsilon, decay_rate)
    q_table_sarsa, rewards_sarsa = train_sarsa(env, episodes, alpha, gamma, max_epsilon, min_epsilon, decay_rate)
    
    env.close()

    smoothing_window = 500
    avg_rewards_q = [np.mean(rewards_q_learning[i:i+smoothing_window]) for i in range(0, len(rewards_q_learning) - smoothing_window + 1, smoothing_window)]
    avg_rewards_sarsa = [np.mean(rewards_sarsa[i:i+smoothing_window]) for i in range(0, len(rewards_sarsa) - smoothing_window + 1, smoothing_window)]
    
    x_axis = range(0, len(rewards_q_learning) - smoothing_window + 1, smoothing_window)

    plt.figure(figsize=(12, 7))
    plt.plot(x_axis, avg_rewards_q, label="Q-learning (off-policy)")
    plt.plot(x_axis, avg_rewards_sarsa, label="SARSA (on-policy)")
    plt.xlabel("Epizody")
    plt.ylabel("Średnia nagroda (wygładzona)")
    plt.title(f"Porównanie skuteczności Q-learning i SARSA w środowisku {env_name}")
    plt.legend()
    plt.grid(True)
    plt.show()

    if np.mean(rewards_q_learning[-1000:]) > np.mean(rewards_sarsa[-1000:]):
        print("\nAgent Q-learning osiągnął lepsze wyniki końcowe.")
        visualize_agent(env_name, q_table_q_learning)
    else:
        print("\nAgent SARSA osiągnął lepsze wyniki końcowe.")
        visualize_agent(env_name, q_table_sarsa)
