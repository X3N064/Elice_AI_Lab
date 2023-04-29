def expected_value(tp, fn, fp, tn):
    tp_value = 7380
    fn_value = 0
    fp_value = -2620
    tn_value = 0	
    
    total = tp + fn + fp + tn
    input_true = tp + fn
    input_false = fp + tn
    if total != 3952:
        s = '''예측 대상인 3,952명 보다 많거나 적은 수가 입력되었습니다.\nConfusion Matrix에 적힌 수를 정확하게 입력해주시기 바랍니다.'''
        raise ValueError('{}'.format(s))
    
    
    
    ev = (tp/total * tp_value) + (fn/total * fn_value) + (fp/total * fp_value) + (tn/total * tn_value)
    revenue = (tp * tp_value) + (fn * fn_value) + (fp * fp_value) + (tn * tn_value)
    print("개발한 모델의 기대손익은 {:,.0f}원 입니다.".format(ev))
    print("개발한 모델을 사용하여 타겟마케팅을 진행했을 때 예상 수익은 {:,.0f}원 입니다.".format(revenue))
