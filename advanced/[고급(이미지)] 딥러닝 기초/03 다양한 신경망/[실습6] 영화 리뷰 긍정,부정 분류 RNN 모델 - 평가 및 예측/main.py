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


# 모델을 구현합니다.
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
model_history = model.fit(X_train, y_train, epochs = 5, verbose = 2)

"""
1. 평가용 데이터를 활용하여 모델을 평가합니다.
   loss와 accuracy를 계산하고 loss, test_acc에 저장합니다.
"""
loss, test_acc = model.evaluate(X_test, y_test, verbose = 0)

"""
2. 평가용 데이터에 대한 예측 결과를 predictions에 저장합니다.
"""
predictions = model.predict(X_test)

# 모델 평가 및 예측 결과를 출력합니다.
print('\nTest Loss : {:.4f} | Test Accuracy : {}'.format(loss, test_acc))
print('예측한 Test Data 클래스 : ',1 if predictions[0]>=0.5 else 0)
