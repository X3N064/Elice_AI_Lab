from model_evaluation import *

def evaluate_expected_value():
    
    '''
    [실습] 모델의 기대손익과 예상 수익 평가하기
    
    1) 이전 실습에서 Confusion Matrix의 결과값을 가져오세요
        --------------------------------------
        |                 |                  |
        |  true_positive  |  false_negative  |
        |                 |                  |
        --------------------------------------
        |                 |                  |
        |  false_positive |  true_negative   |
        |                 |                  |
        --------------------------------------
    
    2) 아래에 각 값을 입력하세요 (총 3,952명을 대상으로 합니다)
    
    '''
    
    # 이전 실습에서 Confusion Matrix의 결과값을 가져와 입력하세요
    true_positive = None
    false_negative = None
    false_positive = None
    true_negative = None
    
    
    # 모델의 기대손익(expected value)과
    # 모델을 사용했을 때의 예상 수익(revenue)을 확인합니다
    expected_value(
                    true_positive, false_negative,
                    false_positive, true_negative
                    )
    
    
    return true_positive,false_negative,false_positive,true_negative
    
if __name__ == "__main__":
    evaluate_expected_value()
