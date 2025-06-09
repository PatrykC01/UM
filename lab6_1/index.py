import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time


print("Uruchamianie demonstracji środowiska CartPole...")
env_cartpole = gym.make("CartPole-v1", render_mode="human")
obs, info = env_cartpole.reset()

for _ in range(100):
    action = env_cartpole.action_space.sample()
    obs, reward, terminated, truncated, info = env_cartpole.step(action)
    done = terminated or truncated
    time.sleep(0.01) 
    if done:
        obs, info = env_cartpole.reset()
env_cartpole.close()
print("Demonstracja CartPole zakończona.")


print("\nRozpoczynanie treningu agenta na FrozenLake z poprawionymi parametrami...")

env_train = gym.make("FrozenLake-v1", is_slippery=False)

n_states = env_train.observation_space.n
n_actions = env_train.action_space.n
Q = np.zeros((n_states, n_actions))

episodes = 10000     
alpha = 0.1           
gamma = 0.99        

epsilon = 1.0             
max_epsilon = 1.0        
min_epsilon = 0.01       
decay_rate = 0.0005       

rewards = []

for episode in range(episodes):
    state, info = env_train.reset()
    done = False
    total_reward = 0
    
    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = env_train.action_space.sample() 
        else:
            action = np.argmax(Q[state])             
        
        next_state, reward, terminated, truncated, info = env_train.step(action)
        done = terminated or truncated
        
        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )
        
        state = next_state
        total_reward += reward
    
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    
    rewards.append(total_reward)

env_train.close()
print("Trening zakończony.")

print("\nTabela Q po treningu:")
print(Q)


print("\nGenerowanie wykresu skuteczności...")

avg_rewards = [np.mean(rewards[i:i+100]) for i in range(0, len(rewards), 100)]
plt.plot(avg_rewards)
plt.xlabel("Kolejne setki epizodów")
plt.ylabel("Średnia nagroda (skuteczność)")
plt.title("Skuteczność agenta Q-learning na FrozenLake")
plt.show()

print("\nUruchamianie wizualizacji wytrenowanego agenta na FrozenLake...")

env_vis = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")
state, info = env_vis.reset()
done = False
max_steps_vis = 100 
steps = 0

while not done and steps < max_steps_vis:

    action = np.argmax(Q[state])
    print(f"Stan: {state}, Wybrana akcja: {action} (0:L, 1:D, 2:P, 3:G)")
    
    next_state, reward, terminated, truncated, info = env_vis.step(action)
    done = terminated or truncated

    state = next_state
    steps += 1
    
    time.sleep(0.4)
    
    if done and reward == 1.0:
        print("\nSukces! Agent dotarł do celu.")
      
        env_vis.render() 
        time.sleep(2)


env_vis.close()
print("Wizualizacja zakończona.")
