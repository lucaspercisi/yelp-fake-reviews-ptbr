# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 20:35:48 2023

@author: lucas
"""

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import preprocess_string
import pandas as pd
import numpy as np
import re
import ast
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import Word2Vec
from sklearn.pipeline import Pipeline
from string import punctuation
from sklearn.metrics import make_scorer, f1_score
from sklearn.preprocessing import StandardScaler

nltk.download('stopwords')
stop_words_pt = set(stopwords.words('portuguese'))

def clean_text(text):
    # Converter para minúsculas
    text = text.lower()

    # Remover pontuações e números
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)

    # Remover stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words_pt])

    return text

def extract_words_from_tagged_content(tagged_content):
    try:
        content_list = ast.literal_eval(tagged_content)
        processed_content = [f'{word.lower()}_{tag}' for word, tag in content_list if word.lower() not in string.punctuation and word.lower() not in stop_words_pt and word.strip() != '']
        return ' '.join(processed_content)
    except ValueError:
        return ""
    
url_dataset = 'https://raw.githubusercontent.com/lucaspercisi/yelp-fake-reviews-ptbr/main/Datasets/portuguese/yelp-fake-reviews-dataset-pt-pos-tagged.csv'
yelp_df = pd.read_csv(url_dataset)

#Contando pontuação
yelp_df['punctuation_count'] = yelp_df['content'].apply(lambda x: len([c for c in str(x) if c in set(punctuation)]))

#Contanto letras em caixa alta
yelp_df['capital_count'] = yelp_df['content'].apply(lambda x: len([c for c in str(x) if c.isupper()]))

#Contando quantidade de palvras
yelp_df['word_count'] = yelp_df['content'].apply(lambda x: len(str(x).split(" ")))

# #limpando conteudo textual
# yelp_df['cleaned_content'] = yelp_df['content'].apply(clean_text)

# #limpando conteudo textual com tag gramtical e convertendo para string
# yelp_df['cleaned_content_tagged'] = yelp_df['content_tagged'].apply(extract_words_from_tagged_content)

# yelp_df_sample = yelp_df.groupby('fake_review').sample(frac=0.50, random_state=42)


# Separando o DataFrame por classe
df_falsos = yelp_df[yelp_df['fake_review'] == True]
df_verdadeiros = yelp_df[yelp_df['fake_review'] == False]

# Contando o número de registros em cada classe
num_falsos = df_falsos.shape[0]
num_verdadeiros = df_verdadeiros.shape[0]

# Amostrando aleatoriamente da classe com mais registros
if num_falsos > num_verdadeiros:
    df_falsos = df_falsos.sample(num_verdadeiros, random_state=42)  # Usando um estado aleatório para reprodutibilidade
else:
    df_verdadeiros = df_verdadeiros.sample(num_falsos, random_state=42)

yelp_df_balanceado = pd.concat([df_falsos, df_verdadeiros])
yelp_df_sample = yelp_df_balanceado.copy()
# yelp_df_sample = yelp_df.copy()

X = yelp_df_sample[['qtd_friends', 'qtd_reviews', 'qtd_photos',	'rating', 'user_has_photo', 'punctuation_count', 'capital_count', 'word_count']]
y = yelp_df_sample['fake_review'].values  

param_grid = {
    # 'Random Forest': {
    #     'classifier': [RandomForestClassifier(n_jobs=-1)],
    #     'classifier__n_estimators': [None, 100, 200, 1000],
    #     'classifier__max_depth': [None, 50, 100],
    #     'classifier__min_samples_split': [0, 1, 2, 10],
    #     'classifier__min_samples_leaf': [0, 1, 10]
    # }
    'Logistic Regression': {
        'classifier': [LogisticRegression(n_jobs=-1)],
        'classifier__C': [0.1, 10, 250, 500],
        'classifier__penalty': ['l2', 'l1'],
        'classifier__solver': ['newton-cg', 'saga', 'lbfgs'],
        'classifier__max_iter': [2000, 5000],
    }

    # 'KNN': {
    #     'classifier': [KNeighborsClassifier(n_jobs=-1)],
    #     'classifier__n_neighbors': [3, 5, 13, 17],
    #     'classifier__weights': ['uniform', 'distance'],
    #     'classifier__metric': ['euclidean', 'manhattan', 'minkowski'],
    #     'classifier__p': [1, 2]  # Parâmetro de potência para a métrica Minkowski
    # }
    # 'XGBoost': {
    #     'classifier': [XGBClassifier(n_jobs=-1)],
    #     'classifier__learning_rate': [0.01, 0.005],
    #     'classifier__n_estimators': [500, 1000],
    #     'classifier__max_depth': [3, 7, 11, 15]
    # }
    # 'SVC': {
    #     'classifier': [SVC()],
    #     'classifier__C': [1, 10, 100, 1000],
    #     'classifier__kernel': ['rbf', 'poly', 'sigmoid'],
    #     'classifier__gamma': ['scale', 'auto'],
    #     'classifier__max_iter': [1000, 2000, 5000] 
    # }
    
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_models = {}

for classifier_name, parameters in param_grid.items():
    print(f"Iniciando GridSearchCV para {classifier_name}")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', parameters['classifier'][0])
    ])

    try:
        grid_search = GridSearchCV(pipeline, parameters, cv=cv, scoring=make_scorer(f1_score), verbose=3)
        grid_search.fit(X, y)

        best_models[classifier_name] = {
            'best_estimator': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }

        print(f"GridSearchCV concluído para {classifier_name}")
        print(f"Melhores parâmetros para {classifier_name}: {grid_search.best_params_}")
        print(f"Melhor score F1 para {classifier_name}: {grid_search.best_score_}")

    except Exception as e:
        print(f"Erro durante GridSearchCV para {classifier_name}: {e}")

