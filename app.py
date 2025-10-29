import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from snake_env import SnakeGame
from dqn_agent import DQN

st.set_page_config(page_title="DQN Snake", layout="centered")
st.title("🐍 DQN スネークゲーム (Colab対応版)")

env = SnakeGame()
state_size = len(env.get_state())
action_size = 3

# モデル読み込み
model = DQN(state_size, 64, action_size)
model.load_state_dict(torch.load("model_snake.pth", map_location="cpu"))
model.eval()

def draw_board(snake, food, width, height):
    board = np.zeros((height, width))
    for (x, y) in snake:
        board[y, x] = 0.5
    fx, fy = food
    board[fy, fx] = 1.0
    fig, ax = plt.subplots()
    ax.imshow(board, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    st.pyplot(fig)

if st.button("▶️ スタート"):
    state = env.reset()
    done = False
    step = 0
    while not done and step < 100:
        action = torch.argmax(model(torch.FloatTensor(state))).item()
        state, reward, done, score = env.step(action)
        st.write(f"Step: {step}, Score: {score}")
        draw_board(env.snake, env.food, env.width, env.height)
        time.sleep(0.3)
        step += 1

    st.success(f"🎉 ゲーム終了！最終スコア: {score}")
