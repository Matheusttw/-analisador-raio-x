# -*- coding: utf-8 -*-
"""
Script completo para classificação de imagens de Raio-X de Tórax (Pneumonia vs. Normal)
Utiliza Transfer Learning com a arquitetura DenseNet121.
"""

# 1. Importação das bibliotecas necessárias
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import cv2

# --- CONFIGURAÇÃO E CONSTANTES ---
# Define os caminhos para os diretórios de dados
# Certifique-se de que a pasta 'chest_xray' está no mesmo diretório que este script
# ou forneça o caminho completo.
BASE_PATH = 'chest_xray'
TRAIN_DIR = os.path.join(BASE_PATH, 'train')
VAL_DIR = os.path.join(BASE_PATH, 'val')
TEST_DIR = os.path.join(BASE_PATH, 'test')

# Parâmetros para o modelo e treinamento
IMG_SIZE = (224, 224) # Tamanho de imagem esperado pela DenseNet
BATCH_SIZE = 32      # Número de imagens por lote
EPOCHS = 15          # Número de épocas para o treinamento
LEARNING_RATE = 0.0001

# --- 2. PRÉ-PROCESSAMENTO E DATA AUGMENTATION ---

# Data Augmentation para o conjunto de treino para aumentar a robustez do modelo
# e evitar overfitting.
train_datagen = ImageDataGenerator(
    rescale=1./255,          # Normaliza os pixels para o intervalo [0, 1]
    rotation_range=20,       # Rotaciona a imagem aleatoriamente em até 20 graus
    width_shift_range=0.1,   # Desloca a imagem horizontalmente
    height_shift_range=0.1,  # Desloca a imagem verticalmente
    shear_range=0.1,         # Aplica cisalhamento
    zoom_range=0.1,          # Aplica zoom
    horizontal_flip=True,    # Inverte a imagem horizontalmente
    fill_mode='nearest'      # Preenche pixels novos com o valor do mais próximo
)

# Para os dados de validação e teste, apenas normalizamos os pixels.
# Não aplicamos augmentation para ter uma avaliação real do desempenho.
test_val_datagen = ImageDataGenerator(rescale=1./255)

# Criação dos geradores de dados
# O gerador lê as imagens dos diretórios, aplica as transformações e as entrega em lotes.
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary' # Problema de duas classes: NORMAL e PNEUMONIA
)

validation_generator = test_val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False # Não é necessário embaralhar os dados de validação
)

test_generator = test_val_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False # Não embaralhar para a avaliação ser consistente
)

# --- 3. CONSTRUÇÃO DO MODELO (TRANSFER LEARNING) ---

def build_model():
    """
    Constrói o modelo de classificação usando Transfer Learning com DenseNet121.
    """
    # Carrega a DenseNet121 pré-treinada na base de dados ImageNet, sem a camada de classificação final.
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    # Congela as camadas da base do modelo para não retreiná-las inicialmente.
    # Elas já contêm features muito úteis aprendidas com milhões de imagens.
    base_model.trainable = False

    # Adiciona nossas próprias camadas de classificação no topo da base congelada.
    inputs = base_model.output
    x = GlobalAveragePooling2D()(inputs) # Camada para achatar as features
    x = Dense(512, activation='relu')(x) # Camada densa com ativação ReLU
    x = Dropout(0.5)(x)                  # Camada de Dropout para regularização (evitar overfitting)
    # Camada de saída com 1 neurônio e ativação sigmoide, ideal para classificação binária.
    # O resultado será uma probabilidade entre 0 (Normal) e 1 (Pneumonia).
    outputs = Dense(1, activation='sigmoid')(x)

    # Cria o modelo final
    model = Model(inputs=base_model.input, outputs=outputs)
    
    # Compila o modelo
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy', # Loss function para classificação binária
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    
    return model

# Constrói o modelo
model = build_model()

# Exibe um resumo da arquitetura do modelo
model.summary()


# --- 4. TREINAMENTO DO MODELO ---

# Callbacks para melhorar o processo de treinamento
# ModelCheckpoint: Salva o modelo com o melhor desempenho na validação.
checkpoint = ModelCheckpoint(
    'best_pneumonia_classifier.h5', # Nome do arquivo para salvar o modelo
    monitor='val_accuracy',       # Métrica a ser monitorada
    save_best_only=True,          # Salva apenas o melhor
    mode='max',                   # 'max' porque queremos maximizar a acurácia
    verbose=1
)

# EarlyStopping: Para o treinamento se a métrica monitorada não melhorar após um certo número de épocas.
early_stopping = EarlyStopping(
    monitor='val_loss', # Monitora a perda na validação
    patience=5,         # Número de épocas sem melhora antes de parar
    restore_best_weights=True, # Restaura os pesos do melhor modelo ao final
    verbose=1
)

# Inicia o treinamento
print("\nIniciando o treinamento do modelo...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[checkpoint, early_stopping]
)


# --- 5. AVALIAÇÃO DO MODELO ---

# Carrega o melhor modelo salvo durante o treinamento
print("\nCarregando o melhor modelo salvo para avaliação...")
best_model = tf.keras.models.load_model('best_pneumonia_classifier.h5')

# Avalia o modelo no conjunto de teste
print("\nAvaliando o modelo no conjunto de teste...")
test_loss, test_acc, test_precision, test_recall = best_model.evaluate(test_generator)
print(f"Acurácia no Teste: {test_acc:.4f}")
print(f"Precisão no Teste: {test_precision:.4f}")
print(f"Recall no Teste: {test_recall:.4f}")

# Gerando predições para criar o relatório de classificação e a matriz de confusão
predictions = best_model.predict(test_generator)
# A saída da sigmoide é uma probabilidade. Convertemos para 0 ou 1 com base em um limiar de 0.5.
predicted_classes = (predictions > 0.5).astype(int).reshape(-1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Relatório de Classificação
print("\nRelatório de Classificação:")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

# Matriz de Confusão
print("\nMatriz de Confusão:")
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Matriz de Confusão')
plt.ylabel('Classe Verdadeira')
plt.xlabel('Classe Prevista')
plt.show()

# Gráficos de Acurácia e Perda durante o treinamento
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Acurácia de Treino')
plt.plot(epochs_range, val_acc, label='Acurácia de Validação')
plt.legend(loc='lower right')
plt.title('Acurácia de Treino e Validação')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Perda de Treino')
plt.plot(epochs_range, val_loss, label='Perda de Validação')
plt.legend(loc='upper right')
plt.title('Perda de Treino e Validação')
plt.show()


# --- 6. FUNÇÃO PARA PREDIÇÃO EM UMA NOVA IMAGEM ---

def predict_image(image_path, model):
    """
    Carrega uma imagem, a pré-processa e retorna a predição do modelo.
    """
    try:
        # Carrega a imagem usando OpenCV
        img = cv2.imread(image_path)
        if img is None:
            print(f"Erro: Não foi possível carregar a imagem em {image_path}")
            return None
            
        # Redimensiona a imagem para o tamanho esperado pelo modelo
        img = cv2.resize(img, IMG_SIZE)
        # Converte a imagem para RGB (OpenCV carrega em BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normaliza os pixels
        img_array = img / 255.0
        # Adiciona uma dimensão de lote (batch dimension)
        img_array = np.expand_dims(img_array, axis=0)

        # Faz a predição
        prediction = model.predict(img_array)[0][0]
        
        # Obtém o rótulo da classe
        if prediction > 0.5:
            return "PNEUMONIA", prediction
        else:
            return "NORMAL", 1 - prediction

    except Exception as e:
        print(f"Ocorreu um erro ao processar a imagem: {e}")
        return None

# Exemplo de uso da função de predição
# Encontre o caminho para uma imagem de teste
# Nota: Este caminho é um exemplo, pode não existir na sua máquina.
# Adapte para uma imagem do seu dataset de teste.
example_image_path = os.path.join(TEST_DIR, 'PNEUMONIA', 'person1_virus_6.jpeg')

if os.path.exists(example_image_path):
    print("\n--- Testando predição em uma nova imagem ---")
    result, confidence = predict_image(example_image_path, best_model)
    print(f'A imagem em "{example_image_path}" foi classificada como: {result}')
    print(f'Confiança da predição: {confidence:.2%}')
else:
    print(f"\nCaminho de imagem de exemplo não encontrado: {example_image_path}. Pulei o teste de predição individual.")