from elice_utils import EliceUtils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

elice_utils = EliceUtils()

x = np.arange(10)
fig, ax = plt.subplots()
ax.plot(
    x, x, label='y=x',
    linestyle='-',
    marker='.',
    color='blue'
)
ax.plot(
    x, x**2, label='y=x^2',
    linestyle='-.',
    marker=',',
    color='red'
)
ax.set_xlabel("x")
ax.set_ylabel("y")

#이미 입력되어 있는 코드의 다양한 속성값들을 변경해 봅시다.
ax.legend(
    loc=6,
    shadow=True,
    fancybox=True,
    borderpad=2
)

# elice에서 그래프를 확인
fig.savefig("plot.png")
elice_utils.send_image("plot.png")

