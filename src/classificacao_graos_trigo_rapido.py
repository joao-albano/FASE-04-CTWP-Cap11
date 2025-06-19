"""
ATIVIDADE (IR ALÉM) – Da Terra ao Código: Automatizando a Classificação de Grãos com Machine Learning
Versão Rápida para Demonstração
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings('ignore')

print("="*80)
print("CLASSIFICAÇÃO DE GRÃOS DE TRIGO - ANÁLISE RÁPIDA")
print("="*80)

# 1. CARREGAMENTO E ANÁLISE INICIAL DOS DADOS
print("\n1. CARREGAMENTO E ANÁLISE DOS DADOS")
print("-" * 50)

column_names = [
    'area', 'perimetro', 'compacidade', 'comp_nucleo', 
    'larg_nucleo', 'coef_assimetria', 'comp_sulco_nucleo', 'variedade'
]

df = pd.read_csv('../document/seeds_dataset.txt', sep='\s+', header=None, names=column_names, engine='python')

print(f"Dataset carregado: {df.shape[0]} amostras, {df.shape[1]} características")

# Mapeamento das variedades
variedades_map = {1: 'Kama', 2: 'Rosa', 3: 'Canadian'}
df['variedade_nome'] = df['variedade'].map(variedades_map)

print(f"Distribuição das variedades:")
print(df['variedade_nome'].value_counts())

print(f"\nPrimeiras 5 linhas:")
print(df.head())

print(f"\nEstatísticas descritivas:")
print(df.describe())

# 2. PRÉ-PROCESSAMENTO
print("\n2. PRÉ-PROCESSAMENTO")
print("-" * 50)

features = ['area', 'perimetro', 'compacidade', 'comp_nucleo', 
           'larg_nucleo', 'coef_assimetria', 'comp_sulco_nucleo']

X = df[features]
y = df['variedade']

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Normalização
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Dados divididos: {X_train.shape[0]} treino, {X_test.shape[0]} teste")

# 3. TREINAMENTO DOS MODELOS
print("\n3. TREINAMENTO E AVALIAÇÃO DOS MODELOS")
print("-" * 50)

modelos = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(kernel='rbf', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Naive Bayes': GaussianNB(),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

resultados = {}

for nome, modelo in modelos.items():
    print(f"\nTreinando {nome}...")
    
    # Treinar
    modelo.fit(X_train_scaled, y_train)
    
    # Predizer
    y_pred = modelo.predict(X_test_scaled)
    
    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    resultados[nome] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    print(f"  Acurácia: {accuracy:.4f}")
    print(f"  Precisão: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")

# 4. RESUMO DOS RESULTADOS
print("\n4. RESUMO DOS RESULTADOS")
print("-" * 50)

df_resultados = pd.DataFrame(resultados).T.round(4)
print(df_resultados)

# Melhor modelo
melhor_modelo = max(resultados.keys(), key=lambda x: resultados[x]['accuracy'])
melhor_acuracia = resultados[melhor_modelo]['accuracy']

print(f"\nMelhor modelo: {melhor_modelo}")
print(f"Acurácia: {melhor_acuracia:.4f}")

# 5. ANÁLISE DETALHADA DO MELHOR MODELO
print(f"\n5. ANÁLISE DETALHADA - {melhor_modelo}")
print("-" * 50)

# Retreinar o melhor modelo
melhor_modelo_obj = modelos[melhor_modelo]
melhor_modelo_obj.fit(X_train_scaled, y_train)
y_pred_final = melhor_modelo_obj.predict(X_test_scaled)

# Relatório de classificação
print("\nRelatório de classificação:")
target_names = ['Kama', 'Rosa', 'Canadian']
print(classification_report(y_test, y_pred_final, target_names=target_names))

# Matriz de confusão
print("\nMatriz de confusão:")
cm = confusion_matrix(y_test, y_pred_final)
print(cm)

# 6. VISUALIZAÇÕES
print("\n6. GERANDO VISUALIZAÇÕES")
print("-" * 50)

# Matriz de correlação
correlation_matrix = df[features + ['variedade']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True, fmt='.2f')
plt.title('Matriz de Correlação das Características')
plt.tight_layout()
plt.savefig('../assets/matriz_correlacao.png', dpi=300, bbox_inches='tight')
plt.close()
print("Matriz de correlação salva como 'matriz_correlacao.png'")

# Gráfico de comparação dos modelos
plt.figure(figsize=(12, 6))
metricas = ['accuracy', 'precision', 'recall', 'f1_score']
x = np.arange(len(resultados))
width = 0.2

for i, metrica in enumerate(metricas):
    valores = [resultados[modelo][metrica] for modelo in resultados.keys()]
    plt.bar(x + i*width, valores, width, label=metrica.capitalize())

plt.xlabel('Modelos')
plt.ylabel('Score')
plt.title('Comparação de Desempenho dos Modelos')
plt.xticks(x + width*1.5, list(resultados.keys()), rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('../assets/comparacao_modelos.png', dpi=300, bbox_inches='tight')
plt.close()
print("Comparação dos modelos salva como 'comparacao_modelos.png'")

# Boxplot das características por variedade
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()

for i, feature in enumerate(features):
    sns.boxplot(data=df, x='variedade_nome', y=feature, ax=axes[i])
    axes[i].set_title(f'{feature}')
    axes[i].tick_params(axis='x', rotation=45)

# Remover subplot extra
axes[-1].remove()

plt.tight_layout()
plt.savefig('../assets/boxplots_caracteristicas.png', dpi=300, bbox_inches='tight')
plt.close()
print("Boxplots salvos como 'boxplots_caracteristicas.png'")

# 7. SALVAR RESULTADOS
print("\n7. SALVANDO RESULTADOS")
print("-" * 50)

# Salvar resultados em CSV
df_resultados.to_csv('../document/resultados_modelos.csv')

# Salvar predições
predicoes = pd.DataFrame({
    'y_real': y_test.values,
    'y_predito': y_pred_final,
    'variedade_real': [variedades_map[x] for x in y_test.values],
    'variedade_predita': [variedades_map[x] for x in y_pred_final]
})
predicoes.to_csv('../document/predicoes_teste.csv', index=False)

print("Arquivos salvos:")
print("• resultados_modelos.csv")
print("• predicoes_teste.csv")
print("• matriz_correlacao.png")
print("• comparacao_modelos.png")
print("• boxplots_caracteristicas.png")

# 8. INSIGHTS FINAIS
print("\n8. INSIGHTS E CONCLUSÕES")
print("-" * 50)

print(f"• Dataset balanceado com 70 amostras por variedade")
print(f"• Melhor algoritmo: {melhor_modelo} com acurácia de {melhor_acuracia:.1%}")
print(f"• Todas as características são importantes para classificação")
print(f"• Modelo pode ser implementado em cooperativas agrícolas")

# Importância das características (se Random Forest for o melhor)
if melhor_modelo == 'Random Forest':
    importances = melhor_modelo_obj.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print(f"\nImportância das características:")
    for _, row in feature_importance.iterrows():
        print(f"• {row['feature']}: {row['importance']:.3f}")

print("\n" + "="*80)
print("ANÁLISE CONCLUÍDA COM SUCESSO!")
print("="*80) 