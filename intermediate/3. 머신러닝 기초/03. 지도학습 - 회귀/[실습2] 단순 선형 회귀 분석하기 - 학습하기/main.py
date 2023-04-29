import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import elice_utils
eu = elice_utils.EliceUtils()


X = [8.70153760, 3.90825773, 1.89362433, 3.28730045, 7.39333004, 2.98984649, 2.25757240, 9.84450732, 9.94589513, 5.48321616]
Y = [5.64413093, 3.75876583, 3.87233310, 4.40990425, 6.43845020, 4.02827829, 2.26105955, 7.15768995, 6.29097441, 5.19692852]

train_X = pd.DataFrame(X, columns=['X'])
train_Y = pd.Series(Y)

"""
1. 모델을 초기화 합니다.
"""
lrmodel = LinearRegression()

"""
2. train_X, train_Y 데이터를 학습합니다.
"""
lrmodel.fit(train_X, train_Y)


# 학습한 결과를 시각화하는 코드입니다.
plt.scatter(X, Y) 
plt.plot([0, 10], [lrmodel.intercept_, 10 * lrmodel.coef_[0] + lrmodel.intercept_], c='r') 
plt.xlim(0, 10) 
plt.ylim(0, 10) 
plt.title('Training Result')
plt.savefig("test.png") 
eu.send_image("test.png")