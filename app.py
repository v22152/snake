import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from snake_env import SnakeGame
from dqn_agent import DQN
import torch

st.title("🐍 DQN スネークゲーム")
st.write("学習済みモデルを使用して自動プレイします。")

env = SnakeGame()
state_size = len(env.get_state())
action_size = 3

model = DQN(state_size, 64, action_size)
model.load_state_dict(torch.load("model_snake.pth", map_location="cpu"))
model.eval()

# 描画関数
def draw_board(snake, food, width, height):
    board = np.zeros((height, width))
    for (x, y) in snake:
        board[y, x] = 0.5
    fx, fy = food
    board[fy, fx] = 1.0
    fig, ax = plt.subplots()
    ax.imshow(board, cmap="Greens")
    ax.set_xticks([])
    ax.set_yticks([])
    st.pyplot(fig)

if st.button("▶️ スタート"):
    state = env.reset()
    done = False
    step = 0
    while not done and step < 200:
        action = torch.argmax(model(torch.FloatTensor(state))).item()
        state, reward, done, score = env.step(action)
        draw_board(env.snake, env.food, env.width, env.height)
        st.write(f"Step: {step}, Score: {score}")
        step += 1
