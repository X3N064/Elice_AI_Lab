import pandas as pd
from elice_utils import EliceUtils

elice_utils = EliceUtils()


# 데이터를 읽어옵니다.
titanic = pd.read_csv('./data/titanic.csv')
print('변환 전: \n',titanic['Sex'].head())

"""
1. replace를 사용하여 male -> 0, female -> 1로 변환합니다.
"""
titanic = titanic.replace({'male':0, 'female':1})

# 변환한 성별 데이터를 출력합니다.
print('\n변환 후: \n',titanic['Sex'].head())
    
