import pandas as pd
from time import sleep
import elice_utils
pd.set_option('display.max_columns', 20)

def load_titanic_dataset():
    print('>>> 타이타닉 데이터셋을 불러옵니다...')
    elice_utils.send_image("./data/image01.png")
    sleep(2)
    print(' ')
    df = pd.read_csv('./data/titanic.csv')
    df = df.drop('PassengerId', axis=1)
    return df
	
def check_missing_values(df):
    if df is None or df.empty:
        print('타이타닉 데이터를 입력해주세요')
        return None
    titanic_null = df.isnull().sum()
    print("\n>> 타이타닉 데이터의 변수별 결측치 비율은 다음과 같습니다.")
    sleep(2)
    for col, val in titanic_null.items():
        print("{} : {:.2f}%".format(col, val/df.shape[0]*100))

def variable_selection(df):
    if df is None or df.empty:
        print('타이타닉 데이터를 입력해주세요')
        return None
    columns = ['Name', 'Ticket', 'Cabin']
    df.drop(columns, axis=1, inplace=True)
    print('[Variable Selection] 불필요한 컬럼을 제거하였습니다...')

def handling_missing_values(df):
    if df is None or df.empty:
        print('타이타닉 데이터를 입력해주세요')
        return None
    age_median = df['Age'].median()
    df['Age'].fillna(age_median, inplace=True)
    print('[Handling Missing Values] 나이(Age)의 결측치를 중앙값(median)으로 채웠습니다...')

def vectorization_sex(df):
    if df is None or df.empty:
        print('타이타닉 데이터를 입력해주세요')
        return None
    try:
        sex_mapping = {'male': 0, 'female': 1}
        df['Sex'].replace(sex_mapping, inplace=True)
        print('[Vectorization] 성별(Age)을 처리하였습니다...')
    except TypeError:
        return df

def vectorization_embarked(df):
    if df is None or df.empty:
        print('타이타닉 데이터를 입력해주세요')
        return None
    try:
        embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}
        df['Embarked'].replace(embarked_mapping, inplace=True)
        print('[Vectorization] 탑승한 곳(Embarked)을 처리하였습니다...')
        return df.copy()
    except TypeError:
        return df
		
def show_result(df):
    sleep(2)
    print('\n\n==========================')
    print('>> 데이터 처리 결과는 다음과 같습니다.')
    print(df.sample(8, random_state=623))