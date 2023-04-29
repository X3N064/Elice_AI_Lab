from elice_utils import EliceUtils
import matplotlib.pyplot as plt

elice_utils = EliceUtils()

def main():
    matplotlib_tutorial()

    
def matplotlib_tutorial():
    '''
    지시사항: data [1,2,3,4]를 정의하여서, plt.plot을 활용해 데이터를 시각화해보세요.
    '''
    data= [1,2,3,4]
    plt.plot(data)
    
    # 엘리스에서 이미지를 출력하기 위해서 필요한 코드입니다.
    plt.savefig("plot.png")
    elice_utils.send_image("plot.png")

if __name__ == "__main__":
    main()