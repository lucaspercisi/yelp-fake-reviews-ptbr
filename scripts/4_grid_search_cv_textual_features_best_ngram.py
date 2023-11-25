# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 00:45:23 2023

@author: lucas
"""

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

yelp_df_sample = yelp_df.groupby('fake_review').sample(frac=0.50, random_state=42)
# yelp_df_sample = yelp_df.copy()

X = yelp_df_sample['cleaned_content']
y = yelp_df_sample['fake_review'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

test_fold = [-1] * len(X_train) + [0] * len(X_test)  # -1 para treino, 0 para teste
ps = PredefinedSplit(test_fold)

# Concatenando os conjuntos de treino e teste
X_combined = np.concatenate([X_train, X_test])
y_combined = np.concatenate([y_train, y_test])



classifiers_params = {
    'Random Forest': {
        'classifier': RandomForestClassifier(),
        'params': {
            'classifier__n_estimators': [200, 400],
            'classifier__max_depth': [50, None],
            'classifier__min_samples_split': [5, 10],
            'classifier__min_samples_leaf': [1]
        }
    },
    'Logistic Regression': {
        'classifier': LogisticRegression(),
        'params': {
            'classifier__max_iter': [100, 1000, 2000],
            'classifier__C': [0.1, 1, 10],
            'classifier__solver': ['newton-cg', 'lbfgs']
        }
    },
    'KNN': {
        'classifier': KNeighborsClassifier(),
        'params': {
            'classifier__n_neighbors': [5, 7, 13],
            'classifier__weights': ['uniform', 'distance'],
            'classifier__metric': ['euclidean']
        }
    },
    'XGBoost': {
        'classifier': XGBClassifier(),
        'params': {
            'classifier__n_estimators': [200, 400],
            'classifier__learning_rate': [0.1, 0.2, 0.4],
            'classifier__max_depth': [5, 7, 13]
        }
    }
}

best_ngrams = {
    'TF-IDF_Random Forest': (3, 3),
    'TF-IDF_Logistic Regression': (1, 1),
    'TF-IDF_KNN': (3, 3),
    'TF-IDF_XGBoost': (1, 3),
    'BoW_Random Forest': (3, 3),
    'BoW_Logistic Regression': (1, 2),
    'BoW_KNN': (1, 1),
    'BoW_XGBoost': (1, 1)
}

vectorizers = {
    'TF-IDF': TfidfVectorizer(use_idf=True),
    'BoW': CountVectorizer()
}

# Repetindo o processo para os outros vetorizadores (TF-IDF e BoW)
# for vect_name in ['TF-IDF', 'BoW']:
#     for clf_name in classifiers_params:
#         ngram_range = best_ngrams[f'{vect_name}_{clf_name}']

#         if vect_name == 'TF-IDF':
#             vectorizer = TfidfVectorizer(ngram_range=ngram_range)
#         else:  # BoW
#             vectorizer = CountVectorizer(ngram_range=ngram_range)

#         X_combined_vect = vectorizer.fit_transform(X_combined)

#         print(f"Iniciando GridSearchCV para {vect_name} com {clf_name}")
#         pipeline = Pipeline([
#             ('classifier', classifiers_params[clf_name]['classifier'])
#         ])

#         grid_search = GridSearchCV(pipeline, classifiers_params[clf_name]['params'], cv=ps, scoring=make_scorer(f1_score), verbose=1)
#         grid_search.fit(X_combined_vect, y_combined)

#         best_params = grid_search.best_params_
#         best_score = grid_search.score(X_combined_vect[len(X_train):], y_test)  # Avaliando no conjunto de teste

#         print(f"Vetorizador: {vect_name}, Classificador: {clf_name}, Melhores parâmetros: {best_params}, Melhor F1 score no teste: {best_score}")
cv = StratifiedKFold(n_splits=5)
best_results  = {}

for vect_name in ['TF-IDF', 'BoW']:
    for clf_name in classifiers_params:
        # Selecionando o ngram_range com base no vetorizador e classificador
        ngram_range = best_ngrams[f'{vect_name}_{clf_name}']

        # Escolhendo o vetorizador apropriado
        if vect_name == 'TF-IDF':
            vectorizer = TfidfVectorizer(ngram_range=ngram_range)
        else:  # BoW
            vectorizer = CountVectorizer(ngram_range=ngram_range)

        print(f"Iniciando GridSearchCV para {vect_name} com {clf_name}")
        pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifiers_params[clf_name]['classifier'])
        ])

        # Configurando e executando o GridSearchCV
        grid_search = GridSearchCV(pipeline, classifiers_params[clf_name]['params'], cv=cv, scoring=make_scorer(f1_score), verbose=1)
        grid_search.fit(X, y)

        # Imprimindo os resultados
        print(f"Vetorizador: {vect_name}, Classificador: {clf_name}, Melhores parâmetros: {grid_search.best_params_}, Melhor F1 score: {grid_search.best_score_}")
        
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
# X_transformed = w2v_vect.transform(X)
X_train_transformed = w2v_vect.transform(X_train)
X_test_transformed = w2v_vect.transform(X_test)
X_combined_transformed = np.vstack((X_train_transformed, X_test_transformed))

# Atualizando os parâmetros no dicionário classifiers_params
for clf_name in classifiers_params:
    params = classifiers_params[clf_name]['params']
    updated_params = {key.replace('classifier__', ''): value for key, value in params.items()}
    classifiers_params[clf_name]['params'] = updated_params

# Verificação (opcional) - para garantir que os parâmetros foram atualizados corretamente
for clf_name, data in classifiers_params.items():
    print(f"{clf_name} parameters: {data['params']}")

# Loop para executar o GridSearchCV para cada classificador
for clf_name, data in classifiers_params.items():
    print(f"Iniciando GridSearchCV para {clf_name} com Word2Vec")

    # Criando o modelo e o GridSearchCV
    classifier = data['classifier']
    grid_search = GridSearchCV(classifier, data['params'], cv=cv, scoring=make_scorer(f1_score), verbose=1)

    # Ajuste do GridSearchCV ao conjunto de dados transformados pelo Word2Vec
    grid_search.fit(X_transformed, y)

    # Exibindo os melhores parâmetros e o melhor score F1
    print(f"Classificador: {clf_name}, Melhores parâmetros: {grid_search.best_params_}, Melhor F1 score: {grid_search.best_score_} com Word2Vec")

# Loop para executar o GridSearchCV para cada classificador
# for clf_name, data in classifiers_params.items():
#     print(f"Iniciando GridSearchCV para {clf_name} com Word2Vec")

#     classifier = data['classifier']
#     grid_search = GridSearchCV(classifier, data['params'], cv=ps, scoring=make_scorer(f1_score), verbose=1)

#     # Ajuste do GridSearchCV ao conjunto de dados transformados pelo Word2Vec
#     grid_search.fit(X_combined_transformed, y_combined)

#     # Melhores parâmetros e score F1 no conjunto de teste
#     best_params = grid_search.best_params_
#     best_score = grid_search.score(X_test_transformed, y_test)

#     print(f"Classificador: {clf_name}, Melhores parâmetros: {best_params}, Melhor F1 score no teste: {best_score}")
    
    