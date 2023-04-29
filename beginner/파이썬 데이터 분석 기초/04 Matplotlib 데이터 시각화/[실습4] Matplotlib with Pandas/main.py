from elice_utils import EliceUtils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

elice_utils = EliceUtils()

# 아래 경로에서 csv파일을 읽어서 df 변수에 저장해보세요.
# 경로: "./data/pokemon.csv"
df = pd.read_csv("./data/pokemon.csv")

# 공격 타입 Type 1, Type 2 중에 Fire 속성이 존재하는 데이터들만 추출해보세요.
fire = df[(df['Type 1'] == 'Fire') | (df['Type 2'] == 'Fire')]
# 공격 타입 Type 1, Type 2 중에 Water 속성이 존재하는 데이터들만 추출해보세요.
water = df[(df['Type 1'] == 'Water') | (df['Type 2'] == 'Water')]

fig, ax = plt.subplots()
# 왼쪽 표를 참고하여 아래 코드를 완성해보세요.
ax.scatter(fire['Attack'], fire['Defense'],
    marker='*', color='red', label='Fire', s=50)
ax.scatter(water['Attack'], water['Defense'],
    marker='.', color='blue', label='Water', s=25)

ax.set_xlabel("Attack")
ax.set_ylabel("Defense")
ax.legend(loc="upper right")

# elice에서 그래프 확인하기
fig.savefig("plot.png")
elice_utils.send_image("plot.png")

