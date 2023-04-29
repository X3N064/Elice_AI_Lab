import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


df = pd.read_csv("data/Advertising.csv")

print('원본 데이터 샘플 :')
print(df.head(),'\n')

# 입력 변수로 사용하지 않는 Unnamed: 0 변수 데이터를 삭제합니다
df = df.drop(columns=['Unnamed: 0'])

"""
1. Sales 변수는 label 데이터로 Y에 저장하고 나머진 X에 저장합니다.
"""
X = df.drop(columns=['Sales'])
Y = df['Sales']

"""
2. 학습용 평가용 데이터로 분리합니다.
"""
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=42)

# 전 처리한 데이터를 출력합니다
print('train_X : ')
print(train_X.head(),'\n')
print('train_Y : ')
print(train_Y.head(),'\n')

print('test_X : ')
print(test_X.head(),'\n')
print('test_Y : ')
print(test_Y.head())
