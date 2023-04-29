import pandas as pd
from elice_utils import EliceUtils

elice_utils = EliceUtils()

"""
1. 표준화를 수행하는 함수를 구현합니다.
"""
def standard(data):
    
    data = (data - data.mean() )/ data.std()
    
    return data
    
# 데이터를 읽어옵니다.
titanic = pd.read_csv('./data/titanic.csv')
print('변환 전: \n',titanic['Fare'].head())

# standard 함수를 사용하여 표준화합니다.
Fare = standard(titanic['Fare'])

# 변환한 Fare 데이터를 출력합니다.
print('\n변환 후: \n',Fare.head())
    
