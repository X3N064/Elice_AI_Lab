from without_domain import *

def data_preparation_without_domain():
    titanic = load_titanic_dataset()
    
    # [데이터 준비 방법]
    
    # 1. Check_missing_values
    
    # 데이터 내의 결측치 비율을 확인합니다.
    # 일반적으로 데이터에 결측치가 있으면 모델이 정상적으로 작동하지 않기 때문에
    # 컬럼 별로 결측치가 얼마나 있는지 비율을 확인하는 것이 중요합니다.
    
    # 2. Variable Selection
    
    # 결측치가 많고 불필요한 컬럼은 제거합니다.
    # 도메인에 따라 다르지만 보통 한 변수내의 데이터 결측치 비율이 50% 이상일 경우,
    # 해당 변수 자체를 제거하는 것이 좋습니다.

    # 3. Handling Missing Values
    
    # 데이터 내에 결측치가 있는 경우 다른 대체값으로 결측치를 채웁니다.
    # 예를 들어, `Age` feature 내의 결측치를 나이의 중앙값(‘median’)으로 채워줍니다.
    
    # 4. Vectorization
    
    # 컴퓨터가 이해할 수 있도록 데이터를 변환합니다.
    # 예를 들어, `sex` feature 값인 ‘female’, 'male’은 사람은 이해할 수 있지만,
    # 컴퓨터는 여성인지 남성인지 알 수 없습니다.
    # 따라서, 컴퓨터가 이해할 수 있도록 male은 0으로, female은 1로 변환합니다.
    
    '''
    # [실습] 
    
    
    아래에 실습 코드를 입력하여
    데이터가 어떻게 처리되는지 확인하세요
    '''
    
    
    
    
    
    # 데이터 처리 결과를 보여줍니다
    show_result(titanic)

if __name__ == "__main__":
	data_preparation_without_domain()