from scipy.io import arff
import pandas as pd
import polars as pl

data, metadata = arff.loadarff(r'C:\Users\User\Projeto_CienciadeDados\trainingdataset_limpo.arff')

df = pd.DataFrame(data)
df2 = pl.DataFrame(data)

#print(df2.schema)
#print(df.head(), df.tail(), df.columns)

#classes = -1 e 1
#target = column Result

#o scipy transformou os atributos categoricos em bytes = b'-1'

#print(df.dtypes) ele transformou em object, converter para int

for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].str.decode('utf-8')


df = df.astype(int)

#print(df.dtypes, df.head(), df.info())

from sklearn.model_selection import train_test_split

x = df.drop(columns=['Result'])
y = df['Result']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=42, stratify=y)

treino_df = x_train.copy()
treino_df['Result'] = y_train

teste_df = x_test.copy()
teste_df['Result'] = y_test

""""
 lembra que tem que dropar as colunas 'Result' pra usar corretamente
 tem que dropar pra x e criar y com a coluna retirada apenas
 tenta manter esse padrao x_train, x_test, y_train, y_test
 seria tipo, 
    x_train = treino_df.drop(columns=['Result'])
    y_train = treino_df['Result']

    x_test = treino_df.drop(columns=['Result'])
    y_test = treino_df['Result'] 

"""

#nao precisa indexar as colunas (adicionaria uma coluna 0...n)

treino_df.to_csv('dataset_treino.csv', index=False)
teste_df.to_csv('dataset_test.csv', index=False)
