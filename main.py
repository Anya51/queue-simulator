import numpy as np
import pandas as pd
import sys
from math import factorial as fact
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

lambda_ = float(sys.argv[1])
x = pd.read_csv(sys.argv[2], index_col=False)
x = x['ritmo'].tolist()

def create_table(data):
    df = pd.DataFrame(data=data)
    return df

def save_table_csv(df, path='output_table'):
    df.to_csv(path + '.csv', encoding='utf-8', index=False)

def save_graph_png(df, x, y, path='output_graph'):
    plt.figure()
    plt.plot(df[x], df[y])
    plt.savefig(path + '.png')

def print_table(df):
    print(df)

def print_graph(df, x, y):
    plt.figure()
    plt.plot(df[x], df[y])
    plt.show()

Poisson = lambda x, lambda_: (np.power(lambda_,x)*np.exp(-1 * lambda_))/fact(x)
Poisson = np.vectorize(Poisson)


y = Poisson(x, lambda_)
path = './output/'  + sys.argv[2].split('/', -1)[-1].split('.', 1)[0]
print(sys.argv[2].split('.', 1)[0])
df = create_table({'ritmo': x, 'frequência': y})
save_table_csv(df, path)
save_graph_png(df, 'ritmo', 'frequência', path)