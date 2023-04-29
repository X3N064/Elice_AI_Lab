from target_marketing import *



def predictive_model_for_target_marketing():
    
    '''
    아래 준비된 9개의 스위치의 값을 변경하면서
    타겟마케팅을 했을 때 응할 것 같은 고객을 예측하기 위한 머신러닝 모델을 만들어보세요.
    
    ※주의!  "#" 뒤에 있는 값만 입력해주세요!
    '''
    switch = {
        'handling_missing_value_1' : 0, # 0, 1, 2, 3, 4, 5, ...
        'handling_missing_value_2' : False, # True or False
        'add_age_categorical' : False,  #True or False
        'add_marketing_info' : False, #True or False
        'add_social_economic_info' : False, #True or False
        'transform_pdays_to_categorical' : False, #True or False
        'transform_duration_to_log_scale' : False, #True or False
        'feature_normalization' : None, # None, 'minmax', 'standard'
        'model_selection' : 'logistic_regression' # ['linear_regression','logistic_regression', 'decision_tree',
                                                  # 'knn', 'ridge_regression', 'k-means', 'lasso_regression',
                                                  # 'naive_bayes', 'neural_network', 'random_forest']    
    }
    
    execute_machine_learning_system(switch)		
    
    return switch


if __name__ == "__main__":
	predictive_model_for_target_marketing()
