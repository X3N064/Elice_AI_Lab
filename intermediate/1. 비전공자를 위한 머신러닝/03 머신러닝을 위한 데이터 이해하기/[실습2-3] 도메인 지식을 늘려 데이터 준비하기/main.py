from with_domain import *


def data_preparation_with_domain():
    titanic = load_titanic_dataset()
    
    '''
    [실습] 지시사항에 맞게 아래에 코드를 입력해주세요
    
    '''
    # 1) Data Preprocessing을 실행시키세요 (아래 한 줄 코드 작성)
    
    
    # 2) Feature Engineering을 실행시키세요 (아래 4개의 변수에 True or False 입력)
    switch = {
        'name' :  None     , # True or False
        'age' :    None   , # True or False
        'age_categorization' :   None   , #True or False
        'familysize':   None     # True or False
    }
    
    feature_engineering(titanic, switch)
    
    
    
    # 3) 결과를 확인하세요
    show_result(titanic)
    
    return switch
	

if __name__ == "__main__":
    data_preparation_with_domain()
