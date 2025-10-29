import streamlit as st
from PIL import Image
import numpy as np
import torch
from snake_env import SnakeGame
from dqn_agent import DQN

st.set_page_config(page_title="DQN Snake", layout="centered")
st.title("🐍 DQN スネークゲーム (Colab対応)")

# セッションステート初期化
if "env" not in st.session_state:
    st.session_state.env = SnakeGame()
    st.session_state.state = st.session_state.env.reset()
    st.session_state.model = DQN(7, 64, 3)
    st.session_state.model.load_state_dict(torch.load("model_snake.pth", map_location="cpu"))
    st.session_state.model.eval()
    st.session_state.done = False
    st.session_state.step = 0

def draw_board(snake, food, width, height):
    board = np.zeros((height, width, 3), dtype=np.uint8)
    for x, y in snake:
        board[y, x] = [0, 255, 0]
    fx, fy = food
    board[fy, fx] = [255, 0, 0]
    return Image.fromarray(board)

if st.button("▶️ 次のステップ"):
    if not st.session_state.done:
        action = torch.argmax(st.session_state.model(torch.FloatTensor(st.session_state.state))).item()
        st.session_state.state, reward, st.session_state.done, score = st.session_state.env.step(action)
        st.session_state.step += 1

img = draw_board(st.session_state.env.snake, st.session_state.env.food, st.session_state.env.width, st.session_state.env.height)
st.image(img, width=300)
st.write(f"Step: {st.session_state.step}, Score: {st.session_state.env.score}")

if st.session_state.done:
    st.success(f"ゲーム終了！最終スコア: {st.session_state.env.score}")
    st.session_state.env = SnakeGame()
    st.session_state.state = st.session_state.env.reset()
    st.session_state.done = False
    st.session_state.step = 0
