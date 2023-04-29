from heart_disease import *


def result_of_heart_disease_prediction():
    # 선택한 3개의 요인이 실제 심장질환에 얼마나 관련성이 있는지 확인합니다.
    # Example: features = ['age', 'sex', 'blood_pressure']
    features = ['age', 'sex', 'blood_pressure']
    check_3_features(features)
    
    return features 

if __name__ == "__main__":
    result_of_heart_disease_prediction()
