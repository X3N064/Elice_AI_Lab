import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  

import warnings
warnings.filterwarnings(action='ignore')

from sklearn.model_selection import train_test_split  
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix 
"""
1. data 폴더 내에 있는 dataset.csv파일을 불러오고, 
   학습용 데이터와 테스트용 데이터를 분리하여 
   반환하는 함수를 구현합니다.
   
   Step01. pandas의 read_csv() 함수를 이용하여 
           data 폴더 내에 있는 dataset.csv파일을
           불러옵니다. 
   
   Step02. 데이터 X와 y를 분리합니다.
           데이터 폴더에 있는 dataset.csv 파일을
           확인하고,
           
           X 데이터와 y 데이터를 분리하여 각 변수에
           저장합니다.
"""
def load_data():
    
    data = pd.read_csv('data/dataset.csv')
    
    X = data.drop('Class', axis=1)
    y = data['Class']
    
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 0)
    print(X, y)
    return train_X, test_X, train_y, test_y
    
"""
2. SVM 모델을 불러오고,
   학습용 데이터에 맞추어 학습시킨 후, 
   테스트 데이터에 대한 예측 결과를 반환하는 함수를
   구현합니다.
   
   Step01. SVM 모델을 정의합니다.
   
   Step02. SVM 모델을 학습용 데이터에 맞추어
           학습시킵니다.
   
   Step03. 학습된 모델을 이용하여 
           테스트 데이터에 대한 예측을 수행합니다. 
"""
def SVM(train_X, test_X, train_y, test_y):
    
    svm = SVC()
    
    svm.fit(train_X, train_y)
    
    pred_y = svm.predict(test_X)
    
    return pred_y
    
# 데이터를 불러오고, 모델 예측 결과를 확인하는 main 함수입니다.
def main():
    
    train_X, test_X, train_y, test_y = load_data()
    
    pred_y = SVM(train_X, test_X, train_y, test_y)
    
    # SVM 분류 결과값을 출력합니다.
    print("\nConfusion matrix : \n",confusion_matrix(test_y,pred_y))  
    print("\nReport : \n",classification_report(test_y,pred_y)) 

if __name__ == "__main__":
    main()
