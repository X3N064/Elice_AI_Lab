import numpy as np
import pandas as pd

# GDP와 인구수 시리즈 값이 들어간 데이터프레임을 생성합니다.
population = pd.Series({'korea': 5180,'japan': 12718,'china': 141500,'usa': 32676})
gdp = pd.Series({'korea': 169320000,'japan': 516700000,'china': 1409250000,'usa': 2041280000})
print("Country DataFrame")
country = pd.DataFrame({"population" : population,"gdp" : gdp})
print(country)


# 데이터프레임에 gdp per capita 칼럼을 추가하고 출력합니다.
gdp_per_capita = country['gdp']/country['population']
country['gdp per capita'] = gdp_per_capita

