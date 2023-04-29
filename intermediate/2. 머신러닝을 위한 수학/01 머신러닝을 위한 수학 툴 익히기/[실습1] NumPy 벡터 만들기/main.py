import numpy as np
from elice_utils import EliceUtils

elice_utils = EliceUtils()

def main():
    print(tutorial_1st())
    print(tutorial_2nd())
    
    

    
def tutorial_1st():
    """
    지시사항 1.
    tutorial_1st() 함수 안에 5번째 값만 1로 가지고 이외의 값은 0을 가지는 
    길이 10의 벡터를 선언하세요.
    """
    A= np.zeros(10)
    A[4] = 1
    return A



    
def tutorial_2nd():
    """
    지시사항 2.
    tutorial_2nd() 함수 안에 10~49의 range를 가지는 벡터를 선언하세요.
    """
    
    B = np.arange(10,50)
    return B


if __name__ == "__main__":
    main()