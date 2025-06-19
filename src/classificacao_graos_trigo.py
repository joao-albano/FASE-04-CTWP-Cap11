"""
ATIVIDADE (IR ALÉM) – Da Terra ao Código: Automatizando a Classificação de Grãos com Machine Learning

Desenvolvido seguindo a metodologia CRISP-DM para classificação de variedades de grãos de trigo.

Autor: Trabalho FIAP Cap 3
Dataset: Seeds Dataset - UCI Machine Learning Repository
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
import warnings

warnings.filterwarnings('ignore')

# Configurar matplotlib para não usar interface gráfica
import matplotlib
matplotlib.use('Agg')

# Configurações para visualização
plt.style.use('default')
sns.set_palette("husl")

print("="*80)
print("CLASSIFICAÇÃO DE GRÃOS DE TRIGO COM MACHINE LEARNING")
print("Metodologia CRISP-DM")
print("="*80)

# ==============================================================================
# 1. ANÁLISE E PRÉ-PROCESSAMENTO DOS DADOS
# ==============================================================================

print("\n1. ANÁLISE E PRÉ-PROCESSAMENTO DOS DADOS")
print("-" * 50)

# 1.1 Carregamento dos dados
print("1.1 Carregando o dataset...")

# Definindo os nomes das colunas
column_names = [
    'area', 'perimetro', 'compacidade', 'comp_nucleo', 
    'larg_nucleo', 'coef_assimetria', 'comp_sulco_nucleo', 'variedade'
]

# Carregando o dataset
df = pd.read_csv('../document/seeds_dataset.txt', sep='\s+', header=None, names=column_names, engine='python')

print(f"Dataset carregado com sucesso!")
print(f"Shape do dataset: {df.shape}")
print(f"Colunas: {list(df.columns)}")

# 1.2 Exibindo as primeiras linhas
print("\n1.2 Primeiras 10 linhas do dataset:")
print(df.head(10))

# 1.3 Informações gerais do dataset
print("\n1.3 Informações gerais do dataset:")
print(df.info())

# 1.4 Verificando valores ausentes
print("\n1.4 Verificação de valores ausentes:")
print(df.isnull().sum())

# 1.5 Mapeamento das variedades
variedades_map = {1: 'Kama', 2: 'Rosa', 3: 'Canadian'}
df['variedade_nome'] = df['variedade'].map(variedades_map)

print("\n1.5 Distribuição das variedades:")
print(df['variedade_nome'].value_counts())
print(f"Distribuição percentual:")
print(df['variedade_nome'].value_counts(normalize=True) * 100)

# 1.6 Estatísticas descritivas
print("\n1.6 Estatísticas descritivas para cada característica:")
stats_desc = df.describe()
print(stats_desc)

# 1.7 Correlação entre características
print("\n1.7 Matriz de correlação:")
correlation_matrix = df.select_dtypes(include=[np.number]).corr()
print(correlation_matrix.round(3))

# ==============================================================================
# 2. VISUALIZAÇÃO DOS DADOS
# ==============================================================================

print("\n2. VISUALIZAÇÃO DOS DADOS")
print("-" * 50)

# 2.1 Configuração das figuras
fig_size = (15, 10)

# 2.2 Histogramas das características
print("2.1 Gerando histogramas das características...")
features = ['area', 'perimetro', 'compacidade', 'comp_nucleo', 
           'larg_nucleo', 'coef_assimetria', 'comp_sulco_nucleo']

plt.figure(figsize=fig_size)
for i, feature in enumerate(features, 1):
    plt.subplot(3, 3, i)
    plt.hist(df[feature], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title(f'Distribuição - {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequência')

plt.tight_layout()
plt.savefig('histogramas_caracteristicas.png', dpi=300, bbox_inches='tight')
plt.close()
print("Histogramas salvos como 'histogramas_caracteristicas.png'")

# 2.3 Boxplots das características por variedade
print("2.2 Gerando boxplots das características por variedade...")
plt.figure(figsize=fig_size)
for i, feature in enumerate(features, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(data=df, x='variedade_nome', y=feature)
    plt.title(f'Boxplot - {feature} por Variedade')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('boxplots_caracteristicas.png', dpi=300, bbox_inches='tight')
plt.close()
print("Boxplots salvos como 'boxplots_caracteristicas.png'")

# 2.4 Matriz de correlação (heatmap)
print("2.3 Gerando heatmap da matriz de correlação...")
plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.2f', cbar_kws={"shrink": .8}, mask=mask)
plt.title('Matriz de Correlação das Características')
plt.tight_layout()
plt.savefig('matriz_correlacao.png', dpi=300, bbox_inches='tight')
plt.close()
print("Matriz de correlação salva como 'matriz_correlacao.png'")

# 2.5 Gráficos de dispersão
print("2.4 Gerando gráficos de dispersão...")
# Selecionando as características mais correlacionadas
scatter_pairs = [
    ('area', 'perimetro'),
    ('comp_nucleo', 'larg_nucleo'),
    ('area', 'comp_nucleo'),
    ('perimetro', 'comp_sulco_nucleo')
]

plt.figure(figsize=fig_size)
for i, (x, y) in enumerate(scatter_pairs, 1):
    plt.subplot(2, 2, i)
    colors = ['red', 'blue', 'green']
    for j, variedade in enumerate(['Kama', 'Rosa', 'Canadian']):
        subset = df[df['variedade_nome'] == variedade]
        plt.scatter(subset[x], subset[y], c=colors[j], label=variedade, alpha=0.7)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f'{x} vs {y}')
    plt.legend()

plt.tight_layout()
plt.savefig('graficos_dispersao.png', dpi=300, bbox_inches='tight')
plt.close()
print("Gráficos de dispersão salvos como 'graficos_dispersao.png'")

# ==============================================================================
# 3. PRÉ-PROCESSAMENTO FINAL
# ==============================================================================

print("\n3. PRÉ-PROCESSAMENTO FINAL")
print("-" * 50)

# 3.1 Preparação dos dados para modelagem
X = df[features]
y = df['variedade']

print(f"Shape dos features (X): {X.shape}")
print(f"Shape do target (y): {y.shape}")

# 3.2 Verificação da necessidade de normalização
print("\n3.1 Análise da necessidade de normalização:")
for feature in features:
    print(f"{feature}: Média = {X[feature].mean():.3f}, Desvio = {X[feature].std():.3f}")

# 3.3 Divisão em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\n3.2 Divisão dos dados:")
print(f"Treino: {X_train.shape[0]} amostras")
print(f"Teste: {X_test.shape[0]} amostras")

# 3.4 Normalização dos dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n3.3 Normalização aplicada com sucesso!")

# ==============================================================================
# 4. IMPLEMENTAÇÃO E COMPARAÇÃO DE ALGORITMOS
# ==============================================================================

print("\n4. IMPLEMENTAÇÃO E COMPARAÇÃO DE ALGORITMOS")
print("-" * 50)

# 4.1 Definição dos algoritmos
algoritmos = {
    'K-Nearest Neighbors (KNN)': KNeighborsClassifier(),
    'Support Vector Machine (SVM)': SVC(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Naive Bayes': GaussianNB(),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

# 4.2 Treinamento e avaliação inicial
resultados_iniciais = {}

print("4.1 Treinamento e avaliação inicial dos modelos:")
print("-" * 30)

for nome, modelo in algoritmos.items():
    print(f"\nTreinando {nome}...")
    
    # Treinar o modelo
    modelo.fit(X_train_scaled, y_train)
    
    # Fazer predições
    y_pred = modelo.predict(X_test_scaled)
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Armazenar resultados
    resultados_iniciais[nome] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'model': modelo
    }
    
    print(f"Acurácia: {accuracy:.4f}")
    print(f"Precisão: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

# 4.3 Resumo dos resultados iniciais
print("\n4.2 Resumo dos resultados iniciais:")
print("-" * 30)
df_resultados = pd.DataFrame(resultados_iniciais).T
df_resultados = df_resultados[['accuracy', 'precision', 'recall', 'f1_score']]
print(df_resultados.round(4))

# 4.4 Identificar o melhor modelo inicial
melhor_modelo_inicial = max(resultados_iniciais.keys(), 
                           key=lambda x: resultados_iniciais[x]['accuracy'])
print(f"\nMelhor modelo inicial: {melhor_modelo_inicial}")
print(f"Acurácia: {resultados_iniciais[melhor_modelo_inicial]['accuracy']:.4f}")

# ==============================================================================
# 5. OTIMIZAÇÃO DOS MODELOS
# ==============================================================================

print("\n5. OTIMIZAÇÃO DOS MODELOS")
print("-" * 50)

# 5.1 Definição dos hiperparâmetros para Grid Search
parametros_grid = {
    'K-Nearest Neighbors (KNN)': {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    'Support Vector Machine (SVM)': {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    },
    'Logistic Regression': {
        'C': [0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs'],
        'penalty': ['l1', 'l2']
    }
}

# 5.2 Otimização dos modelos
resultados_otimizados = {}

print("5.1 Otimização de hiperparâmetros:")
print("-" * 30)

for nome, modelo in algoritmos.items():
    if nome in parametros_grid:
        print(f"\nOtimizando {nome}...")
        
        # Grid Search com validação cruzada
        grid_search = GridSearchCV(
            modelo, parametros_grid[nome], 
            cv=5, scoring='accuracy', n_jobs=-1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        # Melhor modelo
        melhor_modelo = grid_search.best_estimator_
        
        # Fazer predições
        y_pred = melhor_modelo.predict(X_test_scaled)
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Armazenar resultados
        resultados_otimizados[nome] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'best_params': grid_search.best_params_,
            'model': melhor_modelo
        }
        
        print(f"Melhores parâmetros: {grid_search.best_params_}")
        print(f"Acurácia otimizada: {accuracy:.4f}")
        print(f"Melhoria: {accuracy - resultados_iniciais[nome]['accuracy']:.4f}")
    else:
        # Para Naive Bayes (sem hiperparâmetros para otimizar)
        resultados_otimizados[nome] = resultados_iniciais[nome]

# 5.3 Comparação dos resultados
print("\n5.2 Comparação dos resultados (Inicial vs Otimizado):")
print("-" * 30)

comparacao = []
for nome in algoritmos.keys():
    inicial = resultados_iniciais[nome]['accuracy']
    otimizado = resultados_otimizados[nome]['accuracy']
    melhoria = otimizado - inicial
    
    comparacao.append({
        'Modelo': nome,
        'Acurácia Inicial': inicial,
        'Acurácia Otimizada': otimizado,
        'Melhoria': melhoria
    })

df_comparacao = pd.DataFrame(comparacao)
print(df_comparacao.round(4))

# ==============================================================================
# 6. AVALIAÇÃO DETALHADA DO MELHOR MODELO
# ==============================================================================

print("\n6. AVALIAÇÃO DETALHADA DO MELHOR MODELO")
print("-" * 50)

# 6.1 Identificar o melhor modelo otimizado
melhor_modelo_final = max(resultados_otimizados.keys(), 
                         key=lambda x: resultados_otimizados[x]['accuracy'])

print(f"6.1 Melhor modelo final: {melhor_modelo_final}")
print(f"Acurácia: {resultados_otimizados[melhor_modelo_final]['accuracy']:.4f}")

# 6.2 Modelo final e predições
modelo_final = resultados_otimizados[melhor_modelo_final]['model']
y_pred_final = modelo_final.predict(X_test_scaled)

# 6.3 Relatório de classificação detalhado
print("\n6.2 Relatório de classificação detalhado:")
print("-" * 30)
target_names = ['Kama', 'Rosa', 'Canadian']
print(classification_report(y_test, y_pred_final, target_names=target_names))

# 6.4 Matriz de confusão
print("\n6.3 Matriz de confusão:")
cm = confusion_matrix(y_test, y_pred_final)
print(cm)

# 6.5 Visualização da matriz de confusão
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(cmap='Blues', values_format='d')
plt.title(f'Matriz de Confusão - {melhor_modelo_final}')
plt.tight_layout()
plt.savefig('matriz_confusao_melhor_modelo.png', dpi=300, bbox_inches='tight')
plt.close()
print("Matriz de confusão salva como 'matriz_confusao_melhor_modelo.png'")

# 6.6 Validação cruzada do melhor modelo
print("\n6.4 Validação cruzada do melhor modelo:")
cv_scores = cross_val_score(modelo_final, X_train_scaled, y_train, cv=5)
print(f"Scores de validação cruzada: {cv_scores}")
print(f"Média: {cv_scores.mean():.4f}")
print(f"Desvio padrão: {cv_scores.std():.4f}")

# ==============================================================================
# 7. INTERPRETAÇÃO DOS RESULTADOS E INSIGHTS
# ==============================================================================

print("\n7. INTERPRETAÇÃO DOS RESULTADOS E INSIGHTS")
print("-" * 50)

print("7.1 Resumo dos principais resultados:")
print("-" * 30)

print(f"• Dataset analisado: {df.shape[0]} amostras, {len(features)} características")
print(f"• Variedades de trigo: 3 (Kama, Rosa, Canadian)")
print(f"• Distribuição balanceada: 70 amostras por variedade")
print(f"• Melhor algoritmo: {melhor_modelo_final}")
print(f"• Acurácia final: {resultados_otimizados[melhor_modelo_final]['accuracy']:.4f}")

print("\n7.2 Insights sobre as características:")
print("-" * 30)

# Análise da importância das características (para Random Forest)
if 'Random Forest' in resultados_otimizados:
    rf_model = resultados_otimizados['Random Forest']['model']
    if hasattr(rf_model, 'feature_importances_'):
        importances = rf_model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("Importância das características (Random Forest):")
        for _, row in feature_importance.iterrows():
            print(f"• {row['feature']}: {row['importance']:.4f}")

print("\n7.3 Correlações mais significativas:")
print("-" * 30)
# Encontrar correlações mais altas
correlacoes_altas = []
for i in range(len(features)):
    for j in range(i+1, len(features)):
        corr = correlation_matrix.iloc[i, j]
        if abs(corr) > 0.7:
            correlacoes_altas.append((features[i], features[j], corr))

correlacoes_altas.sort(key=lambda x: abs(x[2]), reverse=True)
for feat1, feat2, corr in correlacoes_altas:
    print(f"• {feat1} vs {feat2}: {corr:.3f}")

print("\n7.4 Análise por variedade:")
print("-" * 30)
for variedade in ['Kama', 'Rosa', 'Canadian']:
    subset = df[df['variedade_nome'] == variedade]
    print(f"\n{variedade}:")
    print(f"  • Área média: {subset['area'].mean():.3f}")
    print(f"  • Perímetro médio: {subset['perimetro'].mean():.3f}")
    print(f"  • Compacidade média: {subset['compacidade'].mean():.3f}")

print("\n7.5 Conclusões e recomendações:")
print("-" * 30)
print("• O modelo desenvolvido pode automatizar a classificação de grãos de trigo")
print("• A acurácia obtida permite reduzir significativamente erros humanos")
print("• As características físicas são suficientes para distinção entre variedades")
print("• Recomenda-se implementação em sistema de produção para cooperativas")

# ==============================================================================
# 8. SALVAR RESULTADOS
# ==============================================================================

print("\n8. SALVANDO RESULTADOS")
print("-" * 50)

# Salvar resultados em CSV
df_resultados_finais = pd.DataFrame(resultados_otimizados).T
df_resultados_finais = df_resultados_finais[['accuracy', 'precision', 'recall', 'f1_score']]
df_resultados_finais.to_csv('resultados_modelos.csv')

# Salvar dados de teste e predições para análise posterior
resultados_teste = pd.DataFrame({
    'y_real': y_test.values,
    'y_predito': y_pred_final,
    'variedade_real': [variedades_map[x] for x in y_test.values],
    'variedade_predita': [variedades_map[x] for x in y_pred_final]
})
resultados_teste.to_csv('predicoes_teste.csv', index=False)

print("Arquivos salvos:")
print("• resultados_modelos.csv")
print("• predicoes_teste.csv")
print("• histogramas_caracteristicas.png")
print("• boxplots_caracteristicas.png")
print("• matriz_correlacao.png")
print("• graficos_dispersao.png")
print("• matriz_confusao_melhor_modelo.png")

print("\n" + "="*80)
print("ANÁLISE CONCLUÍDA COM SUCESSO!")
print("="*80)