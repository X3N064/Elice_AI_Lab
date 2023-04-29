import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

columns = ['age','sex','chest_pain','blood_pressure','serum_cholesterol','fasting_blood_sugar',
               'electro','max_heart_rate','angina','st_depression','slope','vessels','thal','diagnosis']
categorical_columns = ['sex', 'chest_pain', 'fasting_blood_sugar', 'electro', 'angina', 'slope', 'vessels']
numeric_columns = ['age', 'blood_pressure', 'serum_cholestoral', 'max_heart_rate', 'st_depression']
target = ['diagnosis']
columns_type = {
    'age': np.float,
    'sex':np.object,
    'chest_pain':np.object,
    'blood_pressure':np.float,
    'serum_cholestoral':np.float,
    'fasting_blood_sugar':np.object,
    'electrocardiographic':np.object,
    'max_heart_rate':np.float,
    'induced_angina':np.object,
    'ST_depression': np.float,
    'slope':np.object,
    'vessels':np.object,
    'thal':np.object,
    'diagnosis':np.int
}

def load_dataset():
	data = pd.read_csv('./data/processed.cleveland.data', dtype=columns_type, names=columns)
	data['diagnosis'].replace(to_replace=[1,2,3,4], value=1, inplace=True)
	#heart.replace(to_replace='?', value=np.nan, inplace=True)
	for c in data.columns[:-1]:
		data[c] = data[c].apply(lambda x: data[data[c]!='?'][c].astype(float).mean() if x == '?' else x)
	data.dropna(axis=0, inplace=True)
	return data #heart

def cleansing_categorical(indices_categorical_columns):
    categorical_pipline =  Pipeline(steps=[
                    ('select', FunctionTransformer(lambda data: data[:, indices_categorical_columns])),
                    ('onehot', OneHotEncoder(sparse=False))
                ])
    return categorical_pipline

def cleansing_numeric(indices_numeric_columns):
    numeric_pipline = Pipeline(steps=[
                    ('select', FunctionTransformer(lambda data: data[:, indices_numeric_columns])),
                    ('scale', StandardScaler())
                ])
    return numeric_pipline

def create_estimator(df, model):
    indices_categorical_columns = df.dtypes == np.object
    indices_numeric_columns = df.dtypes != np.object
    if indices_categorical_columns.sum() != 0 and indices_numeric_columns.sum() != 0:
        estimator = Pipeline(steps=[
            ('cleansing', FeatureUnion(transformer_list=[
                ('categorical', cleansing_categorical(indices_categorical_columns)),
                ('numeric', cleansing_numeric(indices_numeric_columns))
            ])),
            ('modeling', model)
        ])
    elif indices_categorical_columns.sum() !=0 and indices_numeric_columns.sum() == 0:
        estimator = Pipeline(steps=[
            ('cleansing', FeatureUnion(transformer_list=[
                ('categorical', cleansing_categorical(indices_categorical_columns))
            ])),
            ('modeling', model)
        ])
    elif indices_categorical_columns.sum() ==0 and indices_numeric_columns.sum() != 0:
        estimator = Pipeline(steps=[
            ('cleansing', FeatureUnion(transformer_list=[
                ('numeric', cleansing_numeric(indices_numeric_columns))
            ])),
            ('modeling', model)
        ])
    else:
        return None
    return estimator

def scoring_columns(data, using_columns):
    data_x, data_y = data.loc[:, using_columns].copy(), data.iloc[:, -1].copy()
    model = LogisticRegression()
    estimator = create_estimator(data_x, model)
    score = cross_val_score(estimator=estimator, X=data_x, y=data_y, cv=5).mean()
    return score

def is_in_features(features):
	columns = ['age','sex','chest_pain','blood_pressure','serum_cholesterol','fasting_blood_sugar',
               'electro','max_heart_rate','angina','st_depression','slope','vessels','thal','diagnosis']
	return set(features).issubset(columns)

def check_3_features(features):
	in_features = is_in_features(features)
	if not in_features:
		print('Error!! 요인(feature)의 이름을 올바르게 작성했는지 확인해주세요. 철자가 틀렸을 수도 있습니다.')
		return None
	features = list(set(features))
	if len(features) != 3:
		print('Error!! 선택해야 하는 요인의 수는 3개입니다.')
		return None
	heart = load_dataset()
	score = scoring_columns(heart, features)
	print("선택한 요인은 {} 입니다.".format(features))
	print("선택한 요인으로 생성한 머신러닝 모델의 심장질환에 대한 분류 예측 정확도는 다음과 같습니다.")
	print("Accuracy : {0:.2f}%".format(score*100))