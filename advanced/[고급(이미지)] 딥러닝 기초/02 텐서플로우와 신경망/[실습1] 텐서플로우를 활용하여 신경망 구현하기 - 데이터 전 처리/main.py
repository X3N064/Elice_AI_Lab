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

"""
1. Sales 변수는 label 데이터로 Y에 저장하고 나머진 X에 저장합니다.
"""
X = df.drop(columns=['Sales'])
Y = df['Sales']

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.3)

"""
2. 학습용 데이터를 tf.data.Dataset 형태로 변환합니다.
   from_tensor_slices 함수를 사용하여 변환하고 batch를 수행하게 합니다.
"""
train_ds = tf.data.Dataset.from_tensor_slices((train_X.values, train_Y.values))
train_ds = train_ds.shuffle(len(train_X)).batch(batch_size=5)

# 하나의 batch를 뽑아서 feature와 label로 분리합니다.
[(train_features_batch, label_batch)] = train_ds.take(1)

# batch 데이터를 출력합니다.
print('\nFB, TV, Newspaper batch 데이터:\n',train_features_batch)
print('Sales batch 데이터:',label_batch)
