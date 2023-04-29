'''
1. 신호의 총합과 그에 따른 결과 0 또는 1을
   반환하는 함수 perceptron을 완성합니다.
   
   Step01. 입력 받은 값을 이용하여
           신호의 총합을 구합니다.
           
   Step02. 신호의 총합이 0 이상이면 1을, 
           그렇지 않으면 0을 반환하는 활성화 
           함수를 작성합니다.
'''
def perceptron(w, x):
    
    output = w[0] + w[1]*x[0] + w[2]*x[1] + w[3]*x[2] + w[4]*x[3]
    
    if output >= 0 :
        y =1
    else : y = 0
    
    return y, output

# x_1, x_2, x_3, x_4의 값을 순서대로 list 형태로 저장
x = [1,2,3,4]

# w_0, w_1, w_2, w_3, w_4의 값을 순서대로 list 형태로 저장
w = [2, -1, 1, 3, -2]

# 퍼셉트론의 결과를 출력
y, output = perceptron(w,x)

print('output: ', output)
print('y: ', y)