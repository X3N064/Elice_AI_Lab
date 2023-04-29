import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

import elice_utils
eu = elice_utils.EliceUtils()


    
X = [8.70153760, 3.90825773, 1.89362433, 3.28730045, 7.39333004, 2.98984649, 2.25757240, 9.84450732, 9.94589513, 5.48321616]
Y = [5.64413093, 3.75876583, 3.87233310, 4.40990425, 6.43845020, 4.02827829, 2.26105955, 7.15768995, 6.29097441, 5.19692852]

train_X = pd.DataFrame(X, columns=['X'])
train_Y = pd.Series(Y)

# 모델을 트레이닝합니다.
lrmodel = LinearRegression()
lrmodel.fit(train_X, train_Y)

"""
1. train_X에 대해서 예측합니다.
"""
pred_X = lrmodel.predict(train_X)
print('train_X에 대한 예측값 : \n{}\n'.format(pred_X))
print('실제값 : \n{}'.format(train_Y))
