# Analisador de Raio-X de Tórax com Deep Learning e PyQt

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange.svg)
![PyQt5](https://img.shields.io/badge/PyQt5-GUI-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

Uma aplicação de desktop completa para classificar imagens de raios-X de tórax como "Normal" ou "Pneumonia", utilizando um modelo de Deep Learning (CNN) com Transfer Learning e uma interface gráfica intuitiva construída com PyQt5.

## Sumário

* [Visão Geral](#visão-geral)
* [Relevância para Áreas de Tecnologia](#relevância-para-áreas-de-tecnologia)
* [Tecnologias Utilizadas](#tecnologias-utilizadas)
* [Estrutura do Projeto](#estrutura-do-projeto)
* [Instalação e Execução](#instalação-e-execução)
* [Como Treinar o Seu Próprio Modelo](#como-treinar-o-seu-próprio-modelo)
* [Próximos Passos](#próximos-passos)
* [Como Contribuir](#como-contribuir)
* [Licença](#licença)

## Visão Geral

Este projeto demonstra um fluxo de trabalho de Machine Learning de ponta a ponta: desde o pré-processamento de dados e treinamento de um modelo de visão computacional até a sua implantação em uma aplicação de desktop amigável para o usuário final.

### Demonstração da Aplicação

<img width="800" height="629" alt="raiox" src="https://github.com/user-attachments/assets/d9f3b716-4e04-4901-954e-12b4fd2f753b" /> <img width="798" height="629" alt="raiox2" src="https://github.com/user-attachments/assets/47d08101-9e41-451e-b05b-437f05f8e8f6" />



### Principais Funcionalidades

* **Interface Gráfica Intuitiva:** Permite que o usuário carregue facilmente uma imagem de raio-X do seu computador.
* **Modelo de Alta Performance:** Utiliza a arquitetura **DenseNet121** com pesos pré-treinados da ImageNet, otimizada para a tarefa de classificação de imagens médicas.
* **Feedback em Tempo Real:** Exibe o diagnóstico ("NORMAL" ou "PNEUMONIA") e a confiança da predição instantaneamente.
* **Código Modular:** O código é organizado, comentado e separado entre a lógica de treinamento e a aplicação final.

## Relevância para Áreas de Tecnologia

Este projeto é um case prático e relevante para profissionais de diversas áreas:

* **Para Ciência de Dados:** Demonstra um ciclo de vida completo de um projeto de Deep Learning, incluindo data augmentation, transfer learning, treinamento, avaliação de métricas (precisão, recall, matriz de confusão) e a importância de um produto final tangível.

* **Para Análise de Dados:** A análise dos resultados do modelo, a matriz de confusão e as curvas de aprendizado são exemplos práticos de como extrair insights sobre a performance de um classificador e identificar possíveis vieses ou áreas para melhoria.

* **Para Engenharia de Dados:** O script de treinamento utiliza `ImageDataGenerator` do Keras, que simula um pipeline de ingestão e pré-processamento de dados em tempo real. Em um cenário de produção, isso poderia ser escalado com ferramentas como Apache Beam ou Spark para processar grandes volumes de imagens.

* **Para Engenharia de Machine Learning (MLOps):** Este projeto é a base do "M" em MLOps (Machine Learning). O próximo passo seria versionar o modelo (com DVC), empacotar a aplicação com Docker e criar um pipeline de CI/CD para automatizar o re-treinamento e o deploy.

* **Para Programação em Nuvem:**
    * **Treinamento:** O modelo poderia ser treinado em instâncias de GPU na nuvem (AWS EC2, Google AI Platform, Azure ML).
    * **Armazenamento:** O dataset e os modelos podem ser armazenados de forma escalável em serviços como AWS S3, Google Cloud Storage ou Azure Blob Storage.
    * **Deploy:** A lógica de predição poderia ser exposta como uma API (usando FastAPI/Flask) e implantada em serviços serverless como AWS Lambda, Google Cloud Functions ou em contêineres no Google Cloud Run / AWS App Runner.

## Tecnologias Utilizadas

* **Linguagem:** Python 3.9+
* **Machine Learning:** TensorFlow, Keras, Scikit-learn
* **Processamento de Imagem:** OpenCV
* **Interface Gráfica:** PyQt5
* **Análise Numérica e Gráficos:** NumPy, Matplotlib, Seaborn

## Estrutura do Projeto

```
analisador-raio-x/
│
├── .gitignore          # Arquivo para ignorar arquivos desnecessários no Git
├── README.md           # Esta documentação
├── requirements.txt    # Lista de dependências do Python
│
├── train_model.py      # Script para treinar e avaliar o modelo de Deep Learning
├── app.py              # Script da aplicação de desktop com PyQt5
│
└── best_pneumonia_classifier.h5  # (Exemplo) O modelo treinado - baixar separadamente
```

## Instalação e Execução

Siga os passos abaixo para executar a aplicação na sua máquina local.

**1. Clone o Repositório**
```bash
git clone [https://github.com/Matheusttw/analisador-raio-x.git](https://github.com/SEU-USUARIO/analisador-raio-x.git)
cd analisador-raio-x
```

**2. Crie um Ambiente Virtual (Recomendado)**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

**3. Instale as Dependências**
O arquivo `requirements.txt` contém todas as bibliotecas necessárias.
```bash
pip install -r requirements.txt
```
*(Nota: Crie um arquivo `requirements.txt` com o comando `pip freeze > requirements.txt` após instalar as bibliotecas mencionadas).*

**4. Baixe o Modelo Pré-Treinado**
O modelo treinado (`best_pneumonia_classifier.h5`) é muito grande para ser armazenado no Git. Faça o download a partir deste link e coloque-o na pasta raiz do projeto.

**[LINK PARA O DOWNLOAD DO MODELO .h5]** <- *Você pode hospedar o arquivo no Google Drive, Dropbox ou usar o GitHub Releases.*

**5. Execute a Aplicação**
```bash
python app.py
```

## Como Treinar o Seu Próprio Modelo

Se você deseja treinar o modelo do zero:

1.  **Baixe o Dataset:** Siga as instruções no script `train_model.py` para baixar o dataset "Chest X-Ray Images (Pneumonia)" do Kaggle.
2.  **Execute o Script de Treinamento:**
    ```bash
    python train_model.py
    ```
3.  Ao final do processo, o melhor modelo será salvo como `best_pneumonia_classifier.h5`, pronto para ser usado pela aplicação `app.py`.


## Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.
