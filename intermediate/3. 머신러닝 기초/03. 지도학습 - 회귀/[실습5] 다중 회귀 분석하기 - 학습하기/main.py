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

"""
1.  다중 선형 회귀 모델을 초기화 하고 학습합니다
"""
lrmodel = LinearRegression()
lrmodel.fit(train_X, train_Y)

"""
2. 학습된 파라미터 값을 불러옵니다
"""
beta_0 = lrmodel.intercept_ # y절편 (기본 판매량)
beta_1 = lrmodel.coef_[0] # 1번째 변수에 대한 계수 (페이스북)
beta_2 = lrmodel.coef_[1] # 2번째 변수에 대한 계수 (TV)
beta_3 = lrmodel.coef_[2] # 3번째 변수에 대한 계수 (신문)

print("beta_0: %f" % beta_0)
print("beta_1: %f" % beta_1)
print("beta_2: %f" % beta_2)
print("beta_3: %f" % beta_3)