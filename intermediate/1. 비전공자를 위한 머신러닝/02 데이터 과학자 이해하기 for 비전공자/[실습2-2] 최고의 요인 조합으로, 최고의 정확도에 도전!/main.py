from heart_disease import *


def result_of_heart_disease_prediction():
    # 심장질환과 연관성이 높은 핵심 요인을 자유롭게 입력하여 정확도를 높여주세요
    # Example: features = ['age', 'sex', 'blood_pressure', 'serum_cholesterol']
    features = ['age', 'sex', 'blood_pressure', 'serum_cholesterol' ]
    check_features(features)
    
    return features


if __name__ == "__main__":
    result_of_heart_disease_prediction()
