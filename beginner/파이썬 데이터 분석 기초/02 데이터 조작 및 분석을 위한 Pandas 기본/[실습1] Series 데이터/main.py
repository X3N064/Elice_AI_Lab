import numpy as np
import pandas as pd

# 예시) 시리즈 데이터를 만드는 방법.
series = pd.Series([1,2,3,4], index = ['a', 'b', 'c', 'd'], name="Title")
print(series, "\n")


# 국가별 인구 수 시리즈 데이터를 딕셔너리를 사용하여 만들어보세요.


pop = {
    "korea" : 5180,
    "japan" : 12718,
    "china" : 141500,
    "usa" : 32676

}
country = pd.Series(pop)
print(country, "\n")