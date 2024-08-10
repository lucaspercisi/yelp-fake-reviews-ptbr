# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 00:45:23 2023

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
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import PredefinedSplit
from datetime import datetime

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

# yelp_df_sample = yelp_df.groupby('fake_review').sample(frac=0.50, random_state=42)
yelp_df_sample = yelp_df.copy()
X = yelp_df_sample['cleaned_content']
y = yelp_df_sample['fake_review'].values


best_params = {
    'TF-IDF': {
        'Random Forest': {
            'max_depth': 1000,
            'min_samples_leaf': 1,
            'min_samples_split': 3,
            'n_estimators': 500
        },
        'Logistic Regression': {
            'C': 10,
            'solver': 'saga',
            'penalty': 'l2',
            'l1_ratio': 0.75
        },
        'KNN': {
            'metric': 'euclidean',
            'n_neighbors': 3,
            'weights': 'uniform',
            'p': 1
        },
        'XGBoost': {
            'learning_rate': 0.1,
            'max_depth': 7,
            'n_estimators': 400
        },
        'SVC': {
            
        }
    },
    'BoW': {
        'Random Forest': {
            'max_depth': None,
            'min_samples_leaf': 1,
            'min_samples_split': 10,
            'n_estimators': 200
        },
        'Logistic Regression': {
            'C': 10,
            'max_iter': 100,
            'solver': 'newton-cg'
        },
        'KNN': {
            'metric': 'euclidean',
            'n_neighbors': 5,
            'weights': 'uniform'
        },
        'XGBoost': {
            'learning_rate': 0.1,
            'max_depth': 7,
            'n_estimators': 400
        },
        'SVC': {
            
        }
    },
    'Word2Vec': {
        'Random Forest': {
            'max_depth': None,
            'min_samples_leaf': 1,
            'min_samples_split': 10,
            'n_estimators': 400
        },
        'Logistic Regression': {
            'C': 10,
            'max_iter': 100,
            'solver': 'newton-cg'
        },
        'KNN': {
            'metric': 'euclidean',
            'n_neighbors': 5,
            'weights': 'uniform'
        },
        'XGBoost': {
            'learning_rate': 0.1,
            'max_depth': 7,
            'n_estimators': 400
        },
        'SVC': {
            
        }
    }
}

classifiers_tfidf = {
    'Random Forest': RandomForestClassifier(n_jobs=-1, **best_params['TF-IDF']['Random Forest']),
    'Logistic Regression': LogisticRegression(**best_params['TF-IDF']['Logistic Regression']),
    'KNN': KNeighborsClassifier(n_jobs=-1, **best_params['TF-IDF']['KNN']),
    'XGBoost': XGBClassifier(n_jobs=-1, **best_params['TF-IDF']['XGBoost']),
    'SVC': SVC(**best_params['TF-IDF']['SVC'])
}

classifiers_bow = {
    'Random Forest': RandomForestClassifier(n_jobs=-1, **best_params['BoW']['Random Forest']),
    'Logistic Regression': LogisticRegression(**best_params['BoW']['Logistic Regression']),
    'KNN': KNeighborsClassifier(n_jobs=-1, **best_params['BoW']['KNN']),
    'XGBoost': XGBClassifier(n_jobs=-1, **best_params['BoW']['XGBoost']),
    'SVC': SVC(**best_params['BoW']['SVC'])
}

classifiers_word2vec = {
    'Random Forest': RandomForestClassifier(n_jobs=-1, **best_params['Word2Vec']['Random Forest']),
    'Logistic Regression': LogisticRegression(**best_params['Word2Vec']['Logistic Regression']),
    'KNN': KNeighborsClassifier(n_jobs=-1, **best_params['Word2Vec']['KNN']),
    'XGBoost': XGBClassifier(n_jobs=-1, **best_params['Word2Vec']['XGBoost']),
    'SVC': SVC(**best_params['Word2Vec']['SVC'])
}

best_ngrams = {
    'TF-IDF_Random Forest': (1, 3),
    'TF-IDF_Logistic Regression': (1, 1),
    'TF-IDF_KNN': (2, 2),
    'TF-IDF_XGBoost': (1, 3),
    'TF-IDF_SVC': (1, 1),
    'BoW_Random Forest': (1, 3),
    'BoW_Logistic Regression': (2, 3),
    'BoW_KNN': (1, 1),
    'BoW_XGBoost': (3, 3),
    'BoW_SVC': (1, 2)
}

vectorizers = {
    'TF-IDF': TfidfVectorizer(use_idf=True),
    'BoW': CountVectorizer()
}

# Caminho base para salvar os arquivos CSV
caminho_base = 'C:/Users/lucas/Documents/Projetos/yelp-fake-reviews-ptbr/Results/spyder'

# Função para adicionar resultados ao CSV com nome de arquivo dinâmico
def adicionar_ao_csv(dados, caminho_base, vetorizador, classificador):
    # Obtendo a data e hora atuais
    agora = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_arquivo = f"{caminho_base}/resultados_{vetorizador}_{classificador}_{agora}.csv"
    
    df_temp = pd.DataFrame([dados])
    df_temp.to_csv(nome_arquivo, index=False)


# Para vetorizadores TF-IDF e BoW
for vect_name, classifier_dict in {'TF-IDF': classifiers_tfidf, 'BoW': classifiers_bow}.items():
    for clf_name, classifier in classifier_dict.items():
        # Selecionando o ngram_range
        ngram_range = best_ngrams[f'{vect_name}_{clf_name}']
        print(f"Terinando -> {vect_name}, {clf_name}")
        # Escolhendo o vetorizador
        vectorizer = TfidfVectorizer(ngram_range=ngram_range) if vect_name == 'TF-IDF' else CountVectorizer(ngram_range=ngram_range)
        X_vect = vectorizer.fit_transform(X)

        # Calculando o F1 score médio
        scores = cross_val_score(classifier, X_vect, y, cv=5, scoring='f1')
        print(f"{vect_name}, {clf_name}: Média F1 Score = {np.mean(scores)}")
        
        # Adicionando ao CSV
        adicionar_ao_csv({'Vetorizador': vect_name, 'Classificador': clf_name, 'F1 Score': np.mean(scores)}, caminho_base, vect_name, clf_name)


# Função para processar o texto e treinar o modelo Word2Vec
class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=100, window=5, min_count=1):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.word2vec = None
        self.dim = None

    def fit(self, X, y=None):
        sentences = [sentence.split() for sentence in X]
        self.word2vec = Word2Vec(sentences, vector_size=self.vector_size, window=self.window, min_count=self.min_count)
        self.dim = self.word2vec.vector_size
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec.wv[word] for word in words if word in self.word2vec.wv]
                    or [np.zeros(self.dim)], axis=0)
            for words in [sentence.split() for sentence in X]
        ])
    
w2v_vect = Word2VecVectorizer(vector_size=100, window=5, min_count=1)
w2v_vect.fit(X)
X_transformed = w2v_vect.transform(X)

# Para Word2Vec
for clf_name, classifier in classifiers_word2vec.items():
    # Calculando o F1 score médio com os dados transformados pelo Word2Vec
    scores = cross_val_score(classifier, X_transformed, y, cv=5, scoring='f1')
    print(f"Word2Vec, {clf_name}: Média F1 Score = {np.mean(scores)}")
    # Adicionando ao CSV
    adicionar_ao_csv({'Vetorizador': 'Word2Vec', 'Classificador': clf_name, 'F1 Score': np.mean(scores)}, caminho_base, 'Word2Vec', clf_name)
