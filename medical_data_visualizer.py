import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('medical_examination.csv')

# Criação da coluna overweight e normalização dos dados
df['overweight'] = df.apply(lambda row: 1 if row['weight'] / ((row['height']/100) ** 2) > 25 else 0, axis=1)
df['cholesterol'] = df.apply(lambda row: 1 if row['cholesterol'] > 1 else 0, axis=1)
df['gluc'] = df.apply(lambda row: 1 if row['gluc'] > 1 else 0, axis=1)

def draw_cat_plot():
    # Cria df que separa todas as variáveis em relação à coluna cardio, e depois conta a quantidade em cada grupo
    df_cat = df.melt(id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).value_counts().rename(columns={'count': 'total'})

    # Gera e salva o gráfico
    fig = sns.catplot(x='variable', y='total', hue='value', col='cardio', data=df_cat, kind='bar').figure
    fig.savefig('catplot.png')
    return fig


def draw_heat_map():
    # Limpeza dos dados incorretos de acordo com as instruções
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
                 (df['height'] >= df['height'].quantile(0.025)) & 
                 (df['height'] <= df['height'].quantile(0.975)) &
                 (df['weight'] >= df['weight'].quantile(0.025)) &
                 (df['weight'] <= df['weight'].quantile(0.975))]
    
    # Gera a matriz de correlação
    corr = df_heat.corr()

    # Gera gráfico com máscara para o triângulo superior da matriz e salva ele
    fig, ax = plt.subplots(figsize=(12, 8))
    mask = np.triu(corr)
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', vmax=0.3, vmin=-0.1, center=0, square=True, linewidths=0.5)
    fig.savefig('heatmap.png')
    return fig
