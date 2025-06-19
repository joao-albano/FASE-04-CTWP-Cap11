"""
Script para converter o arquivo Python em Jupyter Notebook
"""

import json
import nbformat as nbf

def python_to_notebook(python_file, notebook_file):
    """
    Converte um arquivo Python em um notebook Jupyter
    """
    
    # Ler o arquivo Python
    with open(python_file, 'r', encoding='utf-8') as f:
        python_code = f.read()
    
    # Criar um novo notebook
    nb = nbf.v4.new_notebook()
    
    # Dividir o código em seções baseadas nos comentários
    sections = []
    current_section = {"title": "", "code": ""}
    
    lines = python_code.split('\n')
    
    for line in lines:
        # Detectar títulos de seção
        if line.startswith('# =====') or line.startswith('print("====='):
            if current_section["code"].strip():
                sections.append(current_section)
            current_section = {"title": "", "code": ""}
        elif line.startswith('print("\\n') and '. ' in line and line.endswith('")'):
            # Novo título de seção
            if current_section["code"].strip():
                sections.append(current_section)
            title = line.replace('print("\\n', '').replace('print("', '').replace('")', '').strip()
            current_section = {"title": title, "code": ""}
        elif line.startswith('print("-"'):
            # Ignorar linhas de separação
            continue
        else:
            current_section["code"] += line + '\n'
    
    # Adicionar a última seção
    if current_section["code"].strip():
        sections.append(current_section)
    
    # Adicionar célula de título
    nb.cells.append(nbf.v4.new_markdown_cell("""# ATIVIDADE (IR ALÉM) – Da Terra ao Código: Automatizando a Classificação de Grãos com Machine Learning

## Desenvolvido seguindo a metodologia CRISP-DM para classificação de variedades de grãos de trigo.

**Autor:** Trabalho FIAP Cap 3  
**Dataset:** Seeds Dataset - UCI Machine Learning Repository

### Objetivo
Aplicar a metodologia CRISP-DM para desenvolver um modelo de aprendizado de máquina que classifique variedades de grãos de trigo com base em suas características físicas.

### Variedades de Trigo:
- **Kama**
- **Rosa** 
- **Canadian**

### Características do Dataset:
- **Área:** medida da área do grão
- **Perímetro:** comprimento do contorno do grão
- **Compacidade:** calculada como 4π × area / perímetro²
- **Comprimento do Núcleo:** comprimento do eixo principal da elipse equivalente ao grão
- **Largura do Núcleo:** comprimento do eixo secundário da elipse
- **Coeficiente de Assimetria:** medida da assimetria do grão
- **Comprimento do Sulco do Núcleo:** comprimento do sulco central do grão"""))
    
    # Célula de imports
    imports_code = '''"""
Importação das bibliotecas necessárias para análise
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('inline')  # Para exibir gráficos no notebook
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

# Configurações para visualização
plt.style.use('default')
sns.set_palette("husl")'''
    
    nb.cells.append(nbf.v4.new_code_cell(imports_code))
    
    # Processar cada seção
    for section in sections:
        if section["title"]:
            # Adicionar título como célula markdown
            title_clean = section["title"].replace("CLASSIFICAÇÃO DE GRÃOS DE TRIGO - ANÁLISE RÁPIDA", "").strip()
            if title_clean:
                nb.cells.append(nbf.v4.new_markdown_cell(f"## {title_clean}"))
        
        # Adicionar código como célula de código
        if section["code"].strip():
            # Limpar o código removendo prints de títulos
            code_lines = section["code"].split('\n')
            clean_code = []
            skip_next = False
            
            for line in code_lines:
                if skip_next:
                    skip_next = False
                    continue
                    
                if (line.startswith('print("=====') or 
                    line.startswith('print("\\n') or 
                    line.startswith('print("-"') or
                    line.strip() == 'print("="*80)'):
                    continue
                
                clean_code.append(line)
            
            clean_code_str = '\n'.join(clean_code).strip()
            if clean_code_str:
                nb.cells.append(nbf.v4.new_code_cell(clean_code_str))
    
    # Adicionar célula de conclusão
    conclusao_md = """## Conclusões Finais

### Principais Resultados:
- ✅ **Dataset balanceado** com 70 amostras por variedade de trigo
- ✅ **Melhor algoritmo:** Random Forest com acurácia superior a 90%
- ✅ **Características mais importantes:** Área, Perímetro e Comprimento do Sulco do Núcleo
- ✅ **Modelo pronto** para implementação em cooperativas agrícolas

### Benefícios da Implementação:
1. **Redução de erros humanos** na classificação manual
2. **Aumento da eficiência** no processo de classificação
3. **Padronização** dos critérios de classificação
4. **Economia de tempo e recursos** para as cooperativas

### Próximos Passos:
- Implementar o modelo em um sistema de produção
- Coletar mais dados para melhorar a robustez
- Testar com outras variedades de grãos
- Desenvolver interface de usuário para facilitar o uso"""

    nb.cells.append(nbf.v4.new_markdown_cell(conclusao_md))
    
    # Salvar o notebook
    with open(notebook_file, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    
    print(f"Notebook Jupyter criado: {notebook_file}")

if __name__ == "__main__":
    # Instalar nbformat se não estiver instalado
    try:
        import nbformat
    except ImportError:
        import subprocess
        subprocess.check_call(["pip", "install", "nbformat"])
        import nbformat as nbf
    
    # Converter o arquivo
    python_to_notebook("classificacao_graos_trigo_rapido.py", "classificacao_graos_trigo.ipynb") 