import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("🐍 DQN Snake 学習結果 (Colab用)")

# スコア読み込み
scores = np.load("scores.npy")

st.subheader("各エピソードのスコア")
fig, ax = plt.subplots()
ax.plot(range(1, len(scores)+1), scores, marker='o', linestyle='-')
ax.set_xlabel("エピソード")
ax.set_ylabel("スコア（食べた餌の回数）")
st.pyplot(fig)

st.write(f"平均スコア: {np.mean(scores):.2f}")
st.write(f"最大スコア: {np.max(scores)}")
