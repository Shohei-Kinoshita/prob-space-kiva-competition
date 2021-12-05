import re
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

def replace_str(text):
    # '<br /><br />'の文字列を削除
    text = text.replace('<br /><br />', ' ')
    text = text.replace('<br/><br/>\n\na', ' ')
    text = text.replace('<br/><br/>', ' ')
    text = text.replace('-', ' ')
    text = text.replace(r'\n', ' ') # TODO: 空白スペースに置き換える必要があるか確認
    text = text.replace(r'\n\n', ' ')
    text = text.replace('*', ' ')
    text = text.replace('”', '')
    text = text.replace(',', ' ')
    text = text.replace('(', ' ')
    text = text.replace(')', ' ')
    text = text.replace("'", ' ')
    text = text.replace('  ', ' ')
    text = re.sub(r'[0-9]+', ' ', text)
    text = ' '.join(text.split())
    return text


def replace_str_to_use(text):
    text = text.replace(',', ' ')
    text = text.replace('to', ' ')
    text = re.sub(r'[0-9]+', ' ', text)
    text = text.replace('(', ' ')
    text = text.replace(')', ' ')
    text = text.replace(r'\t', ' ')
    text = text.replace(r'\xa', ' ')
    text = text.replace(r'\t\t\t\t\t\t\t', ' ')
    text = ' '.join(text.split())
    return text


def fill_na_DESCRIPTION_TRANSLATED(df):
    df.loc[72195, 'DESCRIPTION_TRANSLATED'] = df.loc[72195, 'DESCRIPTION']
    return df


def lower_text(text):
    # アルファベットの大文字を小文字に変換
    return text.lower()


def target_encoding_oof(train, test, columns, target_name):
    # アウトオブフォールドでターゲットエンコーディング　(中央値でエンコーディング)
    for col in columns:
        tmp_data = pd.DataFrame({col: train[col], 'target': train[target_name]})
        target_aggregate = tmp_data.groupby(col)['target'].median()
        test[col] = test[col].map(target_aggregate)

        tmp = np.repeat(np.nan, train.shape[0])

        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        for idx_1, idx_2 in kf.split(train):
            target_median = tmp_data.iloc[idx_1].groupby(col)['target'].median()
            tmp[idx_2] = train[col].iloc[idx_2].map(target_median)
        train[col] = tmp
        
        return train, test
    
def label_encoding(train, test, columns):
    # ラベルエンコーディング
    for col in columns:
        le = LabelEncoder()
        le.fit(train[col])
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])
        
    return train, test