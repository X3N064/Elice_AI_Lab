def scoring(answer):
    result = resulting(answer)
    accuracy(result)
    wrong_case(result)
    
def resulting(answer):
    solution = [0, 1, 1, 1, 0, 1, 0, 0, 1, 0]
    result = {'right': [], 'wrong': []}
    for i, (a, s) in enumerate(zip(answer, solution)):
        if a == s:
            result['right'].append(i+1)
        else:
            result['wrong'].append(i+1)
    return result

def accuracy(result):
    acc = len(result['right'])
    print('테스트 한 규칙의 정확도: {} 점'.format(acc*10))

def wrong_case(result):
    print('생존자로 잘못 판별된 case는 {} 번째 case입니다.'.format(result['wrong']))
