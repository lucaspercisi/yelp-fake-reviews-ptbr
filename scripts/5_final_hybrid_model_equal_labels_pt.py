# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 17:19:50 2023

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
yelp_df['word_count'] = yelp_df['content'].apply(lambda x: len(str(x).split(" ")))

#limpando conteudo textual
yelp_df['cleaned_content'] = yelp_df['content'].apply(clean_text)

#limpando conteudo textual com tag gramtical e convertendo para string
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
    df_falsos = df_falsos.sample(num_verdadeiros, random_state=42)  # Usando um estado aleatório para reprodutibilidade
else:
    df_verdadeiros = df_verdadeiros.sample(num_falsos, random_state=42)

yelp_df_balanceado = pd.concat([df_falsos, df_verdadeiros])
yelp_df_sample = yelp_df_balanceado.copy()
X = yelp_df_sample['cleaned_content']
y = yelp_df_sample['fake_review'].values


best_params = {
    'TF-IDF': {
        'Random Forest': {
            'max_depth': None,
            'min_samples_leaf': 1,
            'min_samples_split': 10,
            'n_estimators': 200
        },
        'Logistic Regression': {
            'C': 10,
            'max_iter': 300,
            'solver': 'lbfgs'
        },
        'KNN': {
            'metric': 'euclidean',
            'n_neighbors': 5,
            'weights': 'uniform'
        },
        'XGBoost': {
            'learning_rate': 0.2,
            'max_depth': 5,
            'n_estimators': 200
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
            'learning_rate': 0.2,
            'max_depth': 7,
            'n_estimators': 400
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
            'learning_rate': 0.2,
            'max_depth': 5,
            'n_estimators': 200
        }
    }
}

classifiers_tfidf = {
    'Random Forest': RandomForestClassifier(n_jobs=-1, **best_params['TF-IDF']['Random Forest']),
    'Logistic Regression': LogisticRegression(**best_params['TF-IDF']['Logistic Regression']),
    'KNN': KNeighborsClassifier(n_jobs=-1, **best_params['TF-IDF']['KNN']),
    'XGBoost': XGBClassifier(n_jobs=-1, **best_params['TF-IDF']['XGBoost'])
}

classifiers_bow = {
    'Random Forest': RandomForestClassifier(n_jobs=-1, **best_params['BoW']['Random Forest']),
    'Logistic Regression': LogisticRegression(**best_params['BoW']['Logistic Regression']),
    'KNN': KNeighborsClassifier(n_jobs=-1, **best_params['BoW']['KNN']),
    'XGBoost': XGBClassifier(n_jobs=-1, **best_params['BoW']['XGBoost'])
}

classifiers_word2vec = {
    'Random Forest': RandomForestClassifier(n_jobs=-1, **best_params['Word2Vec']['Random Forest']),
    'Logistic Regression': LogisticRegression(**best_params['Word2Vec']['Logistic Regression']),
    'KNN': KNeighborsClassifier(n_jobs=-1, **best_params['Word2Vec']['KNN']),
    'XGBoost': XGBClassifier(n_jobs=-1, **best_params['Word2Vec']['XGBoost'])
}

colunas_numericas = {
    'XGBoost': ['qtd_friends', 'qtd_reviews', 'qtd_photos'],
    'KNN': ['qtd_reviews', 'qtd_photos'],
    'Random Forest': ['qtd_friends', 'qtd_reviews', 'qtd_photos'],
    'Logistic Regression': ['qtd_friends', 'qtd_reviews', 'word_count']
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

results_df_global = pd.DataFrame(columns=[
    'classifier', 'vectorizer', 'features_used', 'accuracy_mean', 
    'accuracy_variance', 'accuracy_min', 'accuracy_max', 
    'precision_mean', 'precision_variance', 'precision_min', 'precision_max', 
    'recall_mean', 'recall_variance', 'recall_min', 'recall_max', 
    'f1_score_mean', 'f1_score_variance', 'f1_score_min', 'f1_score_max'
])

# scorers = {
#     'accuracy': make_scorer(accuracy_score),
#     'precision': make_scorer(precision_score),
#     'recall': make_scorer(recall_score),
#     'f1_score': make_scorer(f1_score)
# }

# Caminho base para salvar os arquivos CSV
caminho_base = 'C:/Users/lucas/Documents/Projetos/yelp-fake-reviews-ptbr/Results/spyder'

# Função para executar o classificador com um número reduzido de features e salvar os resultados
def run_and_save_results(clf, X, y, classifier_name, vectorizer, results_df, features_useds):
    
    scorers = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1_score': 'f1',
        'roc_auc': 'roc_auc'
    }
    
    # cv_results = cross_val_score(clf, X, y, cv=5, scoring='scorers', return_train_score=False)
    cv_results = cross_validate(clf, X, y, cv=5, scoring=scorers, return_train_score=False)

    # Preparando os resultados
    features_used = ', '.join(features_useds.columns)
    results = {
        'classifier': classifier_name,
        'vectorizer': vectorizer,
        'features_used': features_used,
        'accuracy_mean': np.mean(cv_results['test_accuracy']),
        'accuracy_variance': np.var(cv_results['test_accuracy'], ddof=1),
        'accuracy_min': np.min(cv_results['test_accuracy']),
        'accuracy_max': np.max(cv_results['test_accuracy']),
        'precision_mean': np.mean(cv_results['test_precision']),
        'precision_variance': np.var(cv_results['test_precision'], ddof=1),
        'precision_min': np.min(cv_results['test_precision']),
        'precision_max': np.max(cv_results['test_precision']),
        'recall_mean': np.mean(cv_results['test_recall']),
        'recall_variance': np.var(cv_results['test_recall'], ddof=1),
        'recall_min': np.min(cv_results['test_recall']),
        'recall_max': np.max(cv_results['test_recall']),
        'f1_score_mean': np.mean(cv_results['test_f1_score']),
        'f1_score_variance': np.var(cv_results['test_f1_score'], ddof=1),
        'f1_score_min': np.min(cv_results['test_f1_score']),
        'f1_score_max': np.max(cv_results['test_f1_score']),
        'roc_auc_mean': np.mean(cv_results['test_roc_auc']),
        'roc_auc_variance': np.var(cv_results['test_roc_auc'], ddof=1),
        'roc_auc_min': np.min(cv_results['test_roc_auc']),
        'roc_auc_max': np.max(cv_results['test_roc_auc'])
    }

    # Adicionando ao dataframe de resultados
    updated_results_df = pd.concat([results_df, pd.DataFrame([results])], axis=0, ignore_index=True)

    agora = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_arquivo = f"{caminho_base}/resultados_{vectorizer}_{classifier_name}_{agora}.csv"
    
    updated_results_df.to_csv(nome_arquivo, index=False)
    
    return updated_results_df

# Função para adicionar resultados ao CSV com nome de arquivo dinâmico
def adicionar_ao_csv(dados, caminho_base, vetorizador, classificador):
    # Obtendo a data e hora atuais
    agora = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_arquivo = f"{caminho_base}/resultados_{vetorizador}_{classificador}_{agora}.csv"
    
    df_temp = pd.DataFrame([dados])
    df_temp.to_csv(nome_arquivo, index=False)


# # Para vetorizadores TF-IDF e BoW
# for vect_name, classifier_dict in {'TF-IDF': classifiers_tfidf, 'BoW': classifiers_bow}.items():
#     for clf_name, classifier in classifier_dict.items():
#         ngram_range = best_ngrams[f'{vect_name}_{clf_name}']
#         print(f"Terinando -> {vect_name}, {clf_name}")
        
#         vectorizer = TfidfVectorizer(ngram_range=ngram_range) if vect_name == 'TF-IDF' else CountVectorizer(ngram_range=ngram_range)

#         # Escolhendo as features numéricas apropriadas
#         colunas_a_incluir = colunas_numericas.get(clf_name, [])
#         X_numeric = yelp_df_sample[colunas_a_incluir].values if colunas_a_incluir else None

#         # Transformando o texto e concatenando com as features numéricas
#         X_text = vectorizer.fit_transform(X)
#         X_combined = hstack((X_text, X_numeric)) if X_numeric is not None else X_text

#         # Calculando o F1 score médio
#         scores = cross_val_score(classifier, X_combined, y, cv=5, scoring='f1')
#         print(f"{vect_name}, {clf_name}: Média F1 Score = {np.mean(scores)}")

for vect_name, classifier_dict in {'TF-IDF': classifiers_tfidf, 'BoW': classifiers_bow}.items():
    for clf_name, classifier in classifier_dict.items():
        print(f"Iniciando -> {vect_name}, {clf_name}")

        # Configurando o vetorizador
        ngram_range = best_ngrams[f'{vect_name}_{clf_name}']
        vectorizer = TfidfVectorizer(ngram_range=ngram_range) if vect_name == 'TF-IDF' else CountVectorizer(ngram_range=ngram_range)

        # Escolhendo as features numéricas apropriadas
        colunas_a_incluir = colunas_numericas.get(clf_name, [])
        X_numeric = yelp_df_sample[colunas_a_incluir].values if colunas_a_incluir else None

        # Transformando o texto e concatenando com as features numéricas
        X_text = vectorizer.fit_transform(X)
        X_combined = hstack((X_text, X_numeric)) if X_numeric is not None else X_text

        # Executando o classificador e salvando os resultados
        features_used = pd.DataFrame(X_numeric, columns=colunas_a_incluir)
        results_df_global = run_and_save_results(classifier, X_combined, y, clf_name, vect_name, results_df_global, features_used)

        f1_score_atual = results_df_global[(results_df_global['classifier'] == clf_name) & (results_df_global['vectorizer'] == vect_name)]['f1_score_mean'].iloc[-1]
        print(f"F1 Score para {vect_name} e {clf_name}: {f1_score_atual}")


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
    
# w2v_vect = Word2VecVectorizer(vector_size=100, window=5, min_count=1)
# w2v_vect.fit(X)
# X_transformed = w2v_vect.transform(X)

# # Para Word2Vec, combinando com features numéricas
# for clf_name, classifier in classifiers_word2vec.items():
#     colunas_a_incluir = colunas_numericas.get(clf_name, [])
#     X_numeric = yelp_df_sample[colunas_a_incluir].values if colunas_a_incluir else None
#     X_combined = np.hstack((X_transformed, X_numeric)) if X_numeric is not None else X_transformed

#     scores = cross_val_score(classifier, X_combined, y, cv=5, scoring='f1')
#     print(f"Word2Vec, {clf_name}: Média F1 Score = {np.mean(scores)}")
#     adicionar_ao_csv({'Vetorizador': 'Word2Vec', 'Classificador': clf_name, 'F1 Score': np.mean(scores)}, caminho_base, 'Word2Vec', clf_name)

w2v_vect = Word2VecVectorizer(vector_size=100, window=5, min_count=1)
w2v_vect.fit(X)
X_transformed = w2v_vect.transform(X)

# Para Word2Vec, combinando com features numéricas
for clf_name, classifier in classifiers_word2vec.items():
    print(f"Iniciando Word2Vec, {clf_name}")

    # Escolhendo as features numéricas apropriadas
    colunas_a_incluir = colunas_numericas.get(clf_name, [])
    X_numeric = yelp_df_sample[colunas_a_incluir].values if colunas_a_incluir else None
    X_combined = np.hstack((X_transformed, X_numeric)) if X_numeric is not None else X_transformed

    # Executando o classificador e salvando os resultados
    features_used = pd.DataFrame(X_numeric, columns=colunas_a_incluir)
    results_df_global = run_and_save_results(classifier, X_combined, y, clf_name, 'Word2Vec', results_df_global, features_used)
    
    f1_score_atual = results_df_global[(results_df_global['classifier'] == clf_name) & (results_df_global['vectorizer'] == 'Word2Vec')]['f1_score_mean'].iloc[-1]
    print(f"F1 Score para Word2Vec e {clf_name}: {f1_score_atual}")

