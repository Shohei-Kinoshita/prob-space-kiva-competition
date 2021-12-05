"""LDAで融資の使い道列を10のトピックに分類"""

# ライブラリのインポート
import numpy as np
import pandas as pd
import os
import warnings
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# 各種設定
DICT_DTYPE = {'LOAN_ID': 'str', 'IMAGE_ID': 'str'}
FILE_NAME = 'models/lda_loan_use.pkl'
N_COMP = 10

os.chdir('/Users/kinoshitashouhei/Desktop/competitions/05_Prob_Space/Kiva/')

# 共通のカラム名
from functions.common import *
# 前処理
from functions import preprocessing

def main():
    global DICT_DTYPE, FILE_NAME, N_COMP
    
    warnings.filterwarnings('ignore')
    
    # データの読み込み
    df_train = pd.read_csv('data/train.csv', dtype=DICT_DTYPE)
    df_test = pd.read_csv('data/test.csv', dtype=DICT_DTYPE)
    
    # テキストの正規化
    df_train[COL_LOAN_USE] = df_train[COL_LOAN_USE].apply(preprocessing.replace_str_to_use)
    
    # テキストを小文字へ変換
    df_train[COL_LOAN_USE] = df_train[COL_LOAN_USE].apply(preprocessing.lower_text)
    
    # 単語のカウント行列を作成
    text_vec = CountVectorizer()
    text_vec.fit(df_train[COL_LOAN_USE])
    matrix_word_counts = text_vec.transform(df_train[COL_LOAN_USE])
    
    # 単語のカウント行列から、トピックモデルを学習
    lda = LatentDirichletAllocation(n_components=N_COMP, random_state=0)
    lda.fit(matrix_word_counts)
    
    pickle.dump(lda, open(FILE_NAME, 'wb'))

if __name__ == '__main__':
    main()
