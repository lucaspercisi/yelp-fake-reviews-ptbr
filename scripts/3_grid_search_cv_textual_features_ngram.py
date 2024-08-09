# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 23:36:55 2023

@author: lucas
"""

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import preprocess_string
import pandas as pd
import numpy as np
import re
import ast
import string
import nltk
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from string import punctuation
from sklearn.metrics import make_scorer, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

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


url_dataset = 'https://raw.githubusercontent.com/lucaspercisi/yelp-fake-reviews-ptbr/main/Datasets/portuguese/yelp-fake-reviews-dataset-pt-pos-tagged.csv'
yelp_df = pd.read_csv(url_dataset)

# #Contando pontuação
# yelp_df['punctuation_count'] = yelp_df['content'].apply(lambda x: len([c for c in str(x) if c in set(punctuation)]))

# #Contanto letras em caixa alta
# yelp_df['capital_count'] = yelp_df['content'].apply(lambda x: len([c for c in str(x) if c.isupper()]))

# #Contando quantidade de palvras
# yelp_df['word_count'] = yelp_df['content'].apply(lambda x: len(str(x).split(" ")))

#limpando conteudo textual
yelp_df['cleaned_content'] = yelp_df['content'].apply(clean_text)

#limpando conteudo textual com tag gramtical e convertendo para string
# yelp_df['cleaned_content_tagged'] = yelp_df['content_tagged'].apply(extract_words_from_tagged_content)

# yelp_df_sample = yelp_df.groupby('fake_review').sample(frac=0.3, random_state=42)

# Separando o DataFrame por classe
df_falsos = yelp_df[yelp_df['fake_review'] == True]
df_verdadeiros = yelp_df[yelp_df['fake_review'] == False]

# Contando o número de registros em cada classe
num_falsos = df_falsos.shape[0]
num_verdadeiros = df_verdadeiros.shape[0]

# Amostrando aleatoriamente da classe com mais registros
if num_falsos > num_verdadeiros:
    df_falsos = df_falsos.sample(num_verdadeiros, random_state=42)
else:
    df_verdadeiros = df_verdadeiros.sample(num_falsos, random_state=42)

yelp_df_balanceado = pd.concat([df_falsos, df_verdadeiros])
yelp_df_sample = yelp_df_balanceado.copy()


# yelp_df_sample = yelp_df.copy()

X = yelp_df_sample['cleaned_content']
y = yelp_df_sample['fake_review'].values



classifiers = {
    # 'Random Forest': RandomForestClassifier(),
    # 'Logistic Regression': LogisticRegression(n_jobs=-1),
    # 'KNN': KNeighborsClassifier(n_jobs=-1),
    'XGBoost': XGBClassifier(n_jobs=-1),
    # 'SVC': SVC(),
}

n_grams = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]

vectorizers = {
    'TF-IDF': TfidfVectorizer(use_idf=True),
    'BoW': CountVectorizer()
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_results  = {}

for vect_name, vectorizer in vectorizers.items():
    for clf_name, classifier in classifiers.items():
        print(f"Iniciando GridSearchCV para {vect_name} com {clf_name}")

        pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)
        ])

        # Parâmetros para o GridSearchCV
        params = {
            'vectorizer__ngram_range': n_grams
        }

        grid_search = GridSearchCV(pipeline, params, cv=cv, scoring=make_scorer(f1_score), verbose=3)
        grid_search.fit(X, y)

        best_params = grid_search.best_params_
        best_ngram = best_params['vectorizer__ngram_range']
        best_score = grid_search.best_score_

        best_results[f'{vect_name}_{clf_name}'] = (best_ngram, best_score)
        
        print(f"Melhores parâmetros para {vect_name} com {clf_name}: {best_params}")
        print(f"Melhor n-gram: {best_ngram}, Melhor F1 score: {best_score}")

