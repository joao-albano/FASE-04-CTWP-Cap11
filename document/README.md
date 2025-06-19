# 📂 Diretório DOCUMENT

Este diretório contém toda a documentação e dados do projeto de classificação de grãos de trigo.

## 📁 Arquivos:

### 📊 Dados e Resultados:
- `seeds_dataset.txt` - Dataset original com 210 amostras de grãos
- `seeds.zip` - Dataset compactado
- `resultados_finais_otimizados.csv` - Performance dos modelos após otimização
- `comparacao_inicial_vs_otimizado.csv` - Comparação antes/depois da otimização
- `predicoes_finais.csv` - Predições do melhor modelo no conjunto de teste
- `melhores_hiperparametros.json` - Hiperparâmetros otimizados de cada modelo

### 📋 Documentação Técnica:
- `README.md` - Este arquivo explicativo

## 📈 Resumo dos Resultados:

### 🏆 Melhor Modelo: Random Forest (Otimizado)
- **Acurácia**: 93.7% (+1.6% melhoria)
- **Precisão**: 94.3%
- **Recall**: 93.7%
- **F1-Score**: 93.5%

### 📊 Dataset:
- **210 amostras** balanceadas
- **3 variedades**: Kama, Rosa, Canadian
- **7 características** físicas mensuradas
- **0 valores ausentes**

### 🔧 Otimização:
- **Grid Search** aplicado em 4 modelos
- **2 modelos** com melhoria significativa
- **Random Forest**: +1.6% melhoria
- **Logistic Regression**: +3.2% melhoria 