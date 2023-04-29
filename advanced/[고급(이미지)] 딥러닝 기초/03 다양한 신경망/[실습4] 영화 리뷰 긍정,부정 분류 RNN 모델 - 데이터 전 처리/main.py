import json
import numpy as np
import tensorflow as tf
import data_process
from keras.datasets import imdb
from keras.preprocessing import sequence

import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# 학습용 및 평가용 데이터를 불러오고 샘플 문장을 출력합니다.
X_train, y_train, X_test, y_test = data_process.imdb_data_load()

"""
1. 인덱스로 변환된 X_train, X_test 시퀀스에 패딩을 수행하고 각각 X_train, X_test에 저장합니다.
   시퀀스 최대 길이는 300으로 설정합니다.
"""
X_train = sequence.pad_sequences(X_train, maxlen=300, padding='post')
X_test = sequence.pad_sequences(X_test, maxlen=300, padding='post')

print("\n패딩을 추가한 첫 번째 X_train 데이터 샘플 토큰 인덱스 sequence: \n",X_train[0])