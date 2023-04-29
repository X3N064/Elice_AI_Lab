import json
import numpy as np
import tensorflow as tf
import data_process
from keras.datasets import imdb
from keras.preprocessing import sequence

import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# 동일한 실행 결과 확인을 위한 코드입니다.
np.random.seed(123)
tf.random.set_seed(123)

# 학습용 및 평가용 데이터를 불러오고 샘플 문장을 출력합니다.
X_train, y_train, X_test, y_test = data_process.imdb_data_load()

max_review_length = 300

# 패딩을 수행합니다.
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length, padding='post')
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length, padding='post')


embedding_vector_length = 32

"""
1. 모델을 구현합니다.
   임베딩 레이어 다음으로 `SimpleRNN`을 사용하여 RNN 레이어를 쌓고 노드의 개수는 5개로 설정합니다. 
   Dense 레이어는 0, 1 분류이기에 노드를 1개로 하고 activation을 'sigmoid'로 설정되어 있습니다.
"""
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(1000, embedding_vector_length, input_length = max_review_length),
    tf.keras.layers.SimpleRNN(5),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])

# 모델을 확인합니다.
print(model.summary())

# 학습 방법을 설정합니다.
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# 학습을 수행합니다.
model_history = model.fit(X_train, y_train, epochs = 3, verbose = 2)