from elice_utils import EliceUtils
import pandas as pd

elice_utils = EliceUtils()

def main():
    print(pandas_tutorial())
    
def pandas_tutorial():
    '''
    지시사항: `[2,4,6,8,10]`의 리스트를 pandas의 Series 자료구조로 선언하세요.
    '''
    A = pd.Series([2,4,6,8,10])
    return A


if __name__ == "__main__":
    main()