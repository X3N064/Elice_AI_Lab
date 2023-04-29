import pandas as pd
from elice_utils import EliceUtils

elice_utils = EliceUtils()

    
# 데이터를 읽어옵니다.
titanic = pd.read_csv('./data/titanic.csv')
# 변수 별 데이터 수를 확인하여 결측 값이 어디에 많은지 확인합니다.
print(titanic.info(),'\n')

"""
1. Cabin 변수를 제거합니다.
"""
titanic_1 = titanic.drop(columns=['Cabin'])
# Cabin 변수를 제거 후 결측값이 어디에 남아 있는지 확인합니다.
print('Cabin 변수 제거')
print(titanic_1.info(),'\n')

"""
2. 결측값이 존재하는 샘플 제거합니다.
"""
titanic_2 = titanic_1.dropna()
# 결측값이 존재하는지 확인합니다.
print('결측값이 존재하는 샘플 제거')
print(titanic_2.info())


