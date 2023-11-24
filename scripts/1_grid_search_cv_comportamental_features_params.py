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
yelp_df_sample = yelp_df.copy()

X = yelp_df_sample[['qtd_friends', 'qtd_reviews', 'qtd_photos',	'rating', 'user_has_photo', 'punctuation_count', 'capital_count', 'word_count']]
y = yelp_df_sample['fake_review'].values  

param_grid = {
    'Random Forest': {
        'classifier': [RandomForestClassifier()],
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 100],
        'classifier__min_samples_split': [2, 3, 5],
        'classifier__min_samples_leaf': [1]
    },
    'Logistic Regression': {
        'classifier': [LogisticRegression()],
        'classifier__C': [100, 200, 300],
        'classifier__penalty': ['l2'],
        'classifier__solver': ['newton-cg']
    },
    'KNN': {
        'classifier': [KNeighborsClassifier()],
        'classifier__n_neighbors': [3, 5, 11, 13],
        'classifier__weights': ['uniform'],
        'classifier__metric': ['manhattan']
    },
    'XGBoost': {
        'classifier': [XGBClassifier()],
        'classifier__learning_rate': [0.01, 0.005, 0.001],
        'classifier__n_estimators': [500, 1000, 1500],
        'classifier__max_depth': [7, 11, 15, 27]
    }
}

cv = StratifiedKFold(n_splits=5)

# Criando o loop para cada classificador
best_models = {}
for classifier_name, parameters in param_grid.items():
    print(f"Iniciando GridSearchCV para {classifier_name}")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', parameters['classifier'][0])
    ])

    grid_search = GridSearchCV(pipeline, parameters, cv=cv, scoring=make_scorer(f1_score), verbose=1)
    
    # Substitua X e y pelos seus dados
    grid_search.fit(X, y)

    print(f"GridSearchCV concluído para {classifier_name}")
    print(f"Melhores parâmetros para {classifier_name}: {grid_search.best_params_}")
    print(f"Melhor score F1 para {classifier_name}: {grid_search.best_score_}")

