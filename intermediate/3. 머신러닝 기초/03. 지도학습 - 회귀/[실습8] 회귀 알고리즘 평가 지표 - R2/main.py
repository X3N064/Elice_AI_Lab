import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


# 데이터를 읽고 전 처리합니다
df = pd.read_csv("data/Advertising.csv")
df = df.drop(columns=['Unnamed: 0'])

X = df.drop(columns=['Sales'])
Y = df['Sales']

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=42)


# 다중 선형 회귀 모델을 초기화 하고 학습합니다
lrmodel = LinearRegression()
lrmodel.fit(train_X, train_Y)


# train_X 의 예측값을 계산합니다
pred_train = lrmodel.predict(train_X)

"""
1. train_X 의 R2 값을 계산합니다
"""
R2_train = r2_score(train_Y, pred_train)
print('R2_train : %f' % R2_train)

# test_X 의 예측값을 계산합니다
pred_test = lrmodel.predict(test_X)

"""
2. test_X 의 R2 값을 계산합니다
"""
R2_test = r2_score(test_Y, pred_test)
print('R2_test : %f' % R2_test)
