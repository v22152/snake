from snake_env import SnakeGame
from dqn_agent import DQN
import torch
import random
import numpy as np

env = SnakeGame()
state_size = len(env.get_state())
action_size = 3
model = DQN(state_size, 64, action_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995

memory = []
episodes = 100  # Colab用に少なめ
scores = []

for e in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False
    step = 0
    while not done and step < 200:
        if random.random() < epsilon:
            action = random.randint(0, action_size-1)
        else:
            with torch.no_grad():
                action = torch.argmax(model(torch.FloatTensor(state))).item()
        next_state, reward, done, score = env.step(action)
        memory.append((state, action, reward, next_state, done))
        # 簡易1ステップ学習
        target = reward
        if not done:
            target += gamma * torch.max(model(torch.FloatTensor(next_state))).item()
        output = model(torch.FloatTensor(state))
        target_f = output.clone().detach()
        output[action] = target
        loss = criterion(output, target_f)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        state = next_state
        total_reward += reward
        step += 1

    scores.append(env.score)
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

print("✅ 学習終了")
torch.save(model.state_dict(), "model_snake.pth")
np.save("scores.npy", np.array(scores))
