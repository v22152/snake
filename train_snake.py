from snake_env import SnakeGame
from dqn_agent import DQNAgent
import torch

env = SnakeGame()
state_size = len(env.get_state())
action_size = 3
agent = DQNAgent(state_size, action_size)

episodes = 200
for e in range(episodes):
    state = env.reset()
    total_reward = 0
    for time in range(200):
        action = agent.act(state)
        next_state, reward, done, score = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.replay(32)
        state = next_state
        total_reward += reward
        if done:
            break
    print(f"Episode {e+1}/{episodes} - Score: {score}, Reward: {total_reward:.1f}, Epsilon: {agent.epsilon:.3f}")

torch.save(agent.model.state_dict(), "model_snake.pth")
print("✅ モデル保存完了: model_snake.pth")
