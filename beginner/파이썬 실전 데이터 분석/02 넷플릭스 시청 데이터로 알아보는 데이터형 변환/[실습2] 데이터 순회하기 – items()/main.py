# 사용자가 시청한 작품의 리스트를 저장합니다. 수정하지 마세요. 
user_to_titles = {
    1: [271, 318, 491],
    2: [318, 19, 2980, 475],
    3: [475],
    4: [271, 318, 491, 2980, 19, 318, 475],
    5: [882, 91, 2980, 557, 35],
}
def get_user_to_num_titles(user_to_titles):
    user_to_num_titles = {}
    
    # 아래 함수를 완성하세요.
    for user, id in user_to_titles.items():
        user_to_num_titles[user] = len(id)
    
    
    return user_to_num_titles
    

# 아래 주석을 해제하고 결과를 확인해보세요.  
print(get_user_to_num_titles(user_to_titles))
