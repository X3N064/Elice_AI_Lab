import numpy as np
import pandas as pd

print("Masking & query")
df = pd.DataFrame(np.random.rand(5, 2), columns=["A", "B"])
print(df, "\n")

# 데이터 프레임에서 A컬럼값이 0.5보다 작고 B컬럼 값이 0.3보다 큰값들을 구해봅시다.
# 마스킹 연산을 활용하여 출력해보세요!
print(df[(df['A'] < 0.5) & (df['B'] > 0.3)])

# query 함수를 활용하여 출력해보세요!
print(df.query("A < 0.5 and B > 0.3"))
