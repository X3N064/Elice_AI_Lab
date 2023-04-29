import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# 데이터를 읽고 전 처리합니다
df = pd.read_csv("data/Advertising.csv")
df = df.drop(columns=['Unnamed: 0'])

X = df.drop(columns=['Sales'])
Y = df['Sales']

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=42)


# 다중 선형 회귀 모델을 초기화 하고 학습합니다
lrmodel = LinearRegression()
lrmodel.fit(train_X, train_Y)


print('test_X : ')
print(test_X)

"""
1. test_X에 대해서 예측합니다.
"""
pred_X = lrmodel.predict(test_X)
print('test_X에 대한 예측값 : \n{}\n'.format(pred_X))

# 새로운 데이터 df1을 정의합니다
df1 = pd.DataFrame(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]), columns=['FB', 'TV', 'Newspaper'])
print('df1 : ')
print(df1)

"""
2. df1에 대해서 예측합니다.
"""
pred_df1 = lrmodel.predict(df1)
print('df1에 대한 예측값 : \n{}'.format(pred_df1))