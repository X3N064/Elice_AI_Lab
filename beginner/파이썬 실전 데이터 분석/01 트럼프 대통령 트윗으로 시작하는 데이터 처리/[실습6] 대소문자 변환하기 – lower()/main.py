# 트럼프 대통령의 트윗 세개로 구성된 리스트입니다. 수정하지 마세요.
trump_tweets = [
    "FAKE NEWS - A TOTAL POLITICAL WITCH HUNT!",
    "Any negative polls are fake news, just like the CNN, ABC, NBC polls in the election.",
    "The Fake News media is officially out of control.",
]
 
def lowercase_all_characters(text):
    processed_text = []
    # 아래 코드를 작성하세요.
    for i in text:
        new_text = i.lower()
        processed_text.append(new_text)
    
    
    
    return processed_text


# 아래 주석을 해제하고 결과를 확인해보세요.  
print('\n'.join(lowercase_all_characters(trump_tweets)))