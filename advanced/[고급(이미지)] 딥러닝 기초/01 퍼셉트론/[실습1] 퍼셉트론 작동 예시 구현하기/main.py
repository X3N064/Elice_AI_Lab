# 학습 여부를 예측하는 퍼셉트론 함수
def Perceptron(x_1,x_2):
    
    # 설정한 가중치값을 적용
    w_0 = -5 
    w_1 = -1
    w_2 = 5
    
    # 활성화 함수에 들어갈 값을 계산
    output = w_0+w_1*x_1+w_2*x_2
    
    # 활성화 함수 결과를 계산
    if output < 0:
        y = 0
    else:
        y = 1
    
    return y, output


"""
1. perceptron의 예측 결과가 학습한다:1 이 나오도록
   x_1, x_2에 적절한 값을 입력하세요.
"""
x_1 = 3
x_2 = 5

result, go_out = Perceptron(x_1,x_2)

print("신호의 총합 : %d" % go_out)

if go_out > 0:
    print("학습 여부 : %d\n ==> 학습한다!" % result)
else:
    print("학습 여부 : %d\n ==> 학습하지 않는다!" % result)
