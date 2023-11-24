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

def extract_words_from_tagged_content(tagged_content):
    try:
        content_list = ast.literal_eval(tagged_content)
        processed_content = [f'{word.lower()}_{tag}' for word, tag in content_list if word.lower() not in string.punctuation and word.lower() not in stop_words_pt and word.strip() != '']
        return ' '.join(processed_content)
    except ValueError:
        return ""
    
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
yelp_df['cleaned_content_tagged'] = yelp_df['content_tagged'].apply(extract_words_from_tagged_content)

yelp_df_sample = yelp_df.groupby('fake_review').sample(frac=0.3, random_state=42)
# yelp_df_sample = yelp_df.copy()

X = yelp_df_sample['cleaned_content']
y = yelp_df_sample['fake_review'].values

best_params = {
    'Random Forest': {
        'max_depth': 100,
        'min_samples_leaf': 1,
        'min_samples_split': 5,
        'n_estimators': 100
    },
    'Logistic Regression': {
        'C': 200,
        'penalty': 'l2',
        'solver': 'newton-cg'
    },
    'KNN': {
        'metric': 'manhattan',
        'n_neighbors': 13,
        'weights': 'uniform'
    },
    'XGBoost': {
        'learning_rate': 0.005,
        'max_depth': 7,
        'n_estimators': 1000
    }
}

classifiers = {
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(),
    'KNN': KNeighborsClassifier(),
    'XGBoost': XGBClassifier()
}

n_grams = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]

vectorizers = {
    'TF-IDF': TfidfVectorizer(use_idf=True),
    'BoW': CountVectorizer()
}

cv = StratifiedKFold(n_splits=5)
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

        grid_search = GridSearchCV(pipeline, params, cv=cv, scoring=make_scorer(f1_score), verbose=1)

        # Ajuste do GridSearchCV ao seu conjunto de dados textual
        # Substitua X_text e y pelos seus dados
        grid_search.fit(X, y)

        best_params = grid_search.best_params_
        best_ngram = best_params['vectorizer__ngram_range']
        best_score = grid_search.best_score_

        best_results[f'{vect_name}_{clf_name}'] = (best_ngram, best_score)
        
        print(f"Melhores parâmetros para {vect_name} com {clf_name}: {best_params}")
        print(f"Melhor n-gram: {best_ngram}, Melhor F1 score: {best_score}")

