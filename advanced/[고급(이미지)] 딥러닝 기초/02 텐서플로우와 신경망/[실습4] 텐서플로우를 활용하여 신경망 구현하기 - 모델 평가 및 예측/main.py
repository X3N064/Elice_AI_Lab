import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

np.random.seed(100)
tf.random.set_seed(100)

# 데이터를 DataFrame 형태로 불러 옵니다.
df = pd.read_csv("data/Advertising.csv")

# DataFrame 데이터 샘플 5개를 출력합니다.
print('원본 데이터 샘플 :')
print(df.head(),'\n')

# 의미없는 변수는 삭제합니다.
df = df.drop(columns=['Unnamed: 0'])

X = df.drop(columns=['Sales'])
Y = df['Sales']

# 학습용 테스트용 데이터로 분리합니다.
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.3)

# Dataset 형태로 변환합니다.
train_ds = tf.data.Dataset.from_tensor_slices((train_X.values, train_Y))
train_ds = train_ds.shuffle(len(train_X)).batch(batch_size=5)

# keras를 활용하여 신경망 모델을 생성합니다.
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, input_shape=(3,)),
    tf.keras.layers.Dense(1)
    ])

# 학습용 데이터를 바탕으로 모델의 학습을 수행합니다.
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(train_ds, epochs=100, verbose=2)

"""
1. evaluate 메서드를 사용하여 테스트용 데이터의 loss 값을 계산합니다.
"""
loss = model.evaluate(test_X, test_Y, verbose=0)

"""
2. predict 메서드를 사용하여 테스트용 데이터의 예측값을 계산합니다.
"""
predictions = model.predict(test_X)

# 결과를 출력합니다.
print("테스트 데이터의 Loss 값: ", loss)
for i in range(5):
    print("%d 번째 테스트 데이터의 실제값: %f" % (i, test_Y.iloc[i]))
    print("%d 번째 테스트 데이터의 예측값: %f" % (i, predictions[i][0]))
