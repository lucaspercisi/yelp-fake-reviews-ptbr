# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 20:35:48 2023

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

# yelp_df_sample = yelp_df.groupby('fake_review').sample(frac=0.1, random_state=42)

# Separando o DataFrame por classe
df_falsos = yelp_df[yelp_df['fake_review'] == False]
df_verdadeiros = yelp_df[yelp_df['fake_review'] == True]

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

X = yelp_df_sample[['qtd_friends', 'qtd_reviews', 'qtd_photos',	'rating', 'user_has_photo', 'punctuation_count', 'capital_count', 'word_count']]
y = yelp_df_sample['fake_review'].values  

    


best_params = {
    'Random Forest': {
        'n_estimators': 1000,
        'max_depth': None,
        'min_samples_split': 3,
        'min_samples_leaf': 2
    },
    'Logistic Regression': {
        'C': 500,
        'penalty': 'l2',
        'solver': 'newton-cg',
        'l1_ratio': None
    },
    'KNN': {
        'n_neighbors': 17,
        'weights': 'uniform',
        'metric': 'manhattan',
        'p': 1  # Parâmetro de potência para a métrica Minkowski
    },
    'XGBoost': {
        'learning_rate': 0.01,
        'n_estimators': 1000,
        'max_depth': 7
    },
    'SVC': {
        'C': 10,
        'kernel': 'rbf',
        'gamma': 'scale',
        'max_iter': 10000
    }
}

classifiers = {
    'Random Forest': RandomForestClassifier(n_jobs=-1, **best_params['Random Forest']),
    'Logistic Regression': LogisticRegression(n_jobs=-1, **best_params['Logistic Regression']),
    'KNN': KNeighborsClassifier(n_jobs=-1, **best_params['KNN']),
    'XGBoost': XGBClassifier(n_jobs=-1, **best_params['XGBoost']),
    'SVC': SVC(**best_params['SVC'])
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_features_set = {}

for classifier_name, classifier in classifiers.items():
    print(f"Iniciando seleção de features para {classifier_name}")
    best_f1_score = 0
    best_features = []
    features_to_test = X.columns.tolist()

    while len(features_to_test) > 2:
        feature_importances = np.zeros(len(features_to_test))

        for train_idx, test_idx in cv.split(X[features_to_test], y):
            X_train, X_test = X[features_to_test].iloc[train_idx], X[features_to_test].iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Configurar o classificador
            classifier.fit(X_train, y_train)

            # Calcular a importância de permutação
            perm_importance = permutation_importance(classifier, X_test, y_test, n_repeats=5, random_state=42)
            feature_importances += perm_importance.importances_mean

        # Determinar a feature menos importante
        least_important_feature_idx = np.argmin(feature_importances)
        least_important_feature = features_to_test[least_important_feature_idx]

        # Remover a feature menos importante e reavaliar o desempenho
        features_to_test.remove(least_important_feature)
        mean_f1_score = np.mean(cross_val_score(classifier, X[features_to_test], y, cv=cv, scoring='f1', verbose=2))

        # Atualizar o melhor conjunto de features
        if mean_f1_score > best_f1_score:
            best_f1_score = mean_f1_score
            best_features = features_to_test.copy()

        print(f"Para {classifier_name}, {len(features_to_test)} features restantes. Melhor F1 score até agora: {best_f1_score} -> Atual = {mean_f1_score}")

    # Armazenar o melhor conjunto de features e o score para este classificador
    best_features_set[classifier_name] = (best_features, best_f1_score)
    print(f"Melhores features para {classifier_name} concluído: {best_features} com F1 score: {best_f1_score}")

# Imprimir o melhor conjunto de features e o score para cada classificador
for classifier_name, (features, score) in best_features_set.items():
    print(f"Melhores features finais para {classifier_name}: {features} com F1 score: {score}")
    
