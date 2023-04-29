import numpy as np


def perceptron(w, x):
    
    output = w[1] * x[0] + w[2] * x[1] + w[0]
    
    if output >= 0:
        y = 1
    else:
        y = 0
    
    return y



# Input 데이터
X = [[0,0], [0,1], [1,0], [1,1]]

'''
1. perceptron 함수의 입력으로 들어갈 가중치 값을 입력해주세요.
   순서대로 w_0, w_1, w_2에 해당됩니다.
'''
w = [-2, 1, 1]

# AND Gate를 만족하는지 출력하여 확인
print('perceptron 출력')

for x in X:
    print('Input: ',x[0], x[1], ', Output: ',perceptron(w, x))

