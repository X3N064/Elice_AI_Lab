import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from elice_utils import EliceUtils
elice_utils = EliceUtils()


# sklearn에 저장된 데이터를 불러 옵니다.
X, Y = load_breast_cancer(return_X_y = True)
X = np.array(X)
Y = np.array(Y)

# 학습용 평가용 데이터로 분리합니다
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state = 42)

# 분리된 데이터 정보를 출력합니다
print('학습용 샘플 개수: ',len(train_Y))
print('클래스 0인 학습용 샘플 개수: ',len(train_Y)-sum(train_Y))
print('클래스 1인 학습용 샘플 개수: ',sum(train_Y),'\n')

print('평가용 샘플 개수: ',len(test_Y))
print('클래스 0인 평가용 샘플 개수: ',len(test_Y)-sum(test_Y))
print('클래스 1인 평가용 샘플 개수: ',sum(test_Y),'\n')

# DTmodel에 의사결정나무 모델을 초기화 하고 학습합니다
DTmodel = DecisionTreeClassifier()
DTmodel.fit(train_X, train_Y)


# 예측한 값을 저장합니다
y_pred_train = DTmodel.predict(train_X)
y_pred_test = DTmodel.predict(test_X)

# 혼동 행렬을 계산합니다
cm_train = confusion_matrix(train_Y, y_pred_train)
cm_test = confusion_matrix(test_Y, y_pred_test)
print('train_X Confusion Matrix : \n {}'.format(cm_train))
print('test_X Confusion Matrix : \n {}'.format(cm_test))

"""
1. 정확도를 계산합니다.
"""
acc_train = DTmodel.score(train_X, train_Y)
acc_test = DTmodel.score(test_X, test_Y)

# 정확도를 출력합니다.
print('train_X Accuracy: %f' % (acc_train))
print('test_X Accuracy: %f' % (acc_test))

