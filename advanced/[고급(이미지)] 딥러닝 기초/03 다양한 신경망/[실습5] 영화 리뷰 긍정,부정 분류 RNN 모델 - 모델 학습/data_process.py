import json
import numpy as np
import tensorflow as tf
from keras.datasets import imdb
from keras.preprocessing import sequence

import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# 데이터를 불러오고 전처리하는 함수입니다.

n_of_training_ex = 5000
n_of_testing_ex = 1000

PATH = "./data/"

def imdb_data_load():

    X_train = np.load(PATH + "X_train.npy")[:n_of_training_ex]
    y_train = np.load(PATH + "y_train.npy")[:n_of_training_ex]
    X_test = np.load(PATH + "X_test.npy")[:n_of_testing_ex]
    y_test = np.load(PATH + "y_test.npy")[:n_of_testing_ex]

    # 단어 사전 불러오기
    with open(PATH+"imdb_word_index.json") as f:
        word_index = json.load(f)
    # 인덱스 -> 단어 방식으로 딕셔너리 설정
    inverted_word_index = dict((i, word) for (word, i) in word_index.items())
    # 인덱스를 바탕으로 문장으로 변환
    decoded_sequence = " ".join(inverted_word_index[i] for i in X_train[0])

    
    print("첫 번째 X_train 데이터 샘플 문장: \n",decoded_sequence)
    print("\n첫 번째 X_train 데이터 샘플 토큰 인덱스 sequence: \n",X_train[0])
    print("첫 번째 X_train 데이터 샘플 토큰 시퀀스 길이: ", len(X_train[0]))
    print("첫 번째 y_train 데이터: ",y_train[0])
    
    return X_train, y_train, X_test, y_test