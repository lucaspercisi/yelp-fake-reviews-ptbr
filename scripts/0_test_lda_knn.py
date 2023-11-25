# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 10:38:17 2023

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
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import PredefinedSplit
from scipy.sparse import hstack
from datetime import datetime
from sklearn.model_selection import cross_validate
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split


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

#limpando conteudo textual
yelp_df['cleaned_content'] = yelp_df['content'].apply(clean_text)

# Pré-processamento (ajuste conforme necessário)
# df['processed_content'] = df['content'].apply(preprocess_function)

# Vetorização
vectorizer = CountVectorizer(max_df=0.95, min_df=2)
X = vectorizer.fit_transform(yelp_df['content'])

print('iniciando LDA')
# LDA
n_topics = 10  # Escolha um número adequado de tópicos
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
X_topics = lda.fit_transform(X)

print('iniciando train test split')
# Divisão dos dados para treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X_topics, yelp_df['fake_review'], test_size=0.3, random_state=42)

print('iniciando LDA')
# Treinamento do KNN
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)

# Avaliação
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Acurácia: {accuracy}")
print(f"Acurácia: {f1}")

from sklearn.metrics import classification_report
f1_scores = classification_report(y_test, y_pred, average=None)
f1_false = f1_scores[0]  # F1 score para a classe 'False'
f1_true = f1_scores[1]  # F1 score para a classe 'True'





