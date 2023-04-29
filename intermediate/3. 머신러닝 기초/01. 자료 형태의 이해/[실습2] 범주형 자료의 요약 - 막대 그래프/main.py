from elice_utils import EliceUtils
import matplotlib.pyplot as plt
elice_utils = EliceUtils()    
# 술자리 참석 상대도수 데이터 
labels = ['A', 'B', 'C', 'D', 'E']
ratio = [4,3,2,2,1]
    
#막대 그래프
fig, ax = plt.subplots()

"""
1. 막대 그래프를 만드는 코드를 작성해 주세요
"""
plt.bar(labels,ratio)

# 출력에 필요한 코드
plt.show()
fig.savefig("bar_plot.png")
elice_utils.send_image("bar_plot.png")
