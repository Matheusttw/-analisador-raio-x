# -*- coding: utf-8 -*-
"""
Aplicação de Desktop para Classificação de Raio-X de Tórax
Utiliza PyQt5 para a interface gráfica e TensorFlow/Keras para o modelo de IA.

Autor: Gemini
"""

# 1. Importações necessárias
import sys
import os
import numpy as np
import cv2
import tensorflow as tf
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                             QVBoxLayout, QWidget, QFileDialog, QHBoxLayout)
from PyQt5.QtGui import QPixmap, QFont, QIcon
from PyQt5.QtCore import Qt, QSize

# --- CONFIGURAÇÕES E CONSTANTES ---
IMG_SIZE = (224, 224)
MODEL_PATH = 'best_pneumonia_classifier.h5'

# Classe principal da nossa aplicação
class XRayClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # --- Configuração da Janela Principal ---
        self.setWindowTitle("Analisador de Raio-X de Tórax")
        self.setGeometry(100, 100, 800, 600) # (x, y, largura, altura)
        self.setMinimumSize(QSize(600, 500))

        # --- Carregamento do Modelo ---
        self.model = self.load_model()
        
        # --- Inicialização da Interface Gráfica ---
        self.initUI()

    def load_model(self):
        """Carrega o modelo Keras treinado. Retorna None se o arquivo não for encontrado."""
        if not os.path.exists(MODEL_PATH):
            # Se o modelo não for encontrado, trataremos isso na UI.
            return None
        try:
            print("Carregando o modelo treinado...")
            model = tf.keras.models.load_model(MODEL_PATH)
            print("Modelo carregado com sucesso!")
            return model
        except Exception as e:
            print(f"Erro ao carregar o modelo: {e}")
            return None

    def initUI(self):
        """Cria e organiza todos os widgets da interface gráfica."""
        # Widget central para conter todos os outros elementos
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        
        # Layout principal (vertical)
        main_layout = QVBoxLayout(central_widget)
        
        # --- Rótulo para Exibição da Imagem ---
        self.image_label = QLabel("Nenhuma imagem selecionada.\n\nClique em 'Carregar Raio-X' para começar.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFont(QFont('Arial', 14))
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 10px;
                background-color: #f0f0f0;
                color: #555;
            }
        """)
        self.image_label.setMinimumHeight(400)
        main_layout.addWidget(self.image_label)
        
        # --- Layout para os Resultados (horizontal) ---
        result_layout = QHBoxLayout()
        
        # Rótulo para o resultado da predição
        self.result_label = QLabel("Diagnóstico: -")
        self.result_label.setFont(QFont('Arial', 16, QFont.Bold))
        self.result_label.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(self.result_label)
        
        # Rótulo para a confiança da predição
        self.confidence_label = QLabel("Confiança: -")
        self.confidence_label.setFont(QFont('Arial', 16))
        self.confidence_label.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(self.confidence_label)
        
        main_layout.addLayout(result_layout)

        # --- Botão para Carregar Imagem ---
        self.load_button = QPushButton("Carregar Raio-X")
        self.load_button.setFont(QFont('Arial', 14))
        self.load_button.setMinimumHeight(50)
        self.load_button.setStyleSheet("""
            QPushButton {
                background-color: #007BFF;
                color: white;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """)
        # Conecta o clique do botão à função `load_image`
        self.load_button.clicked.connect(self.load_image)
        main_layout.addWidget(self.load_button)

        # --- Verifica se o modelo foi carregado ---
        if self.model is None:
            self.show_model_error()

    def show_model_error(self):
        """Exibe uma mensagem de erro se o arquivo do modelo não for encontrado."""
        self.result_label.setText("ERRO")
        self.result_label.setStyleSheet("color: red;")
        self.confidence_label.setText(f"Arquivo '{MODEL_PATH}' não encontrado!")
        self.load_button.setEnabled(False) # Desabilita o botão
        self.load_button.setText("Modelo não encontrado")

    def load_image(self):
        """Abre uma caixa de diálogo para o usuário selecionar um arquivo de imagem."""
        # Abre o seletor de arquivos, filtrando por tipos de imagem comuns
        file_path, _ = QFileDialog.getOpenFileName(self, "Selecione uma imagem de Raio-X", "", 
                                                   "Imagens (*.png *.jpg *.jpeg *.bmp)")
        
        if file_path:
            # Se um arquivo foi selecionado, exibe a imagem e faz a predição
            pixmap = QPixmap(file_path)
            # Redimensiona a imagem para caber no rótulo, mantendo a proporção
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), 
                                                     Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
            self.predict_image(file_path)

    def predict_image(self, image_path):
        """Pré-processa a imagem e usa o modelo para fazer a predição."""
        try:
            # Carrega e pré-processa a imagem da mesma forma que no treinamento
            img = cv2.imread(image_path)
            img = cv2.resize(img, IMG_SIZE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Garante que está em RGB
            img_array = img / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Faz a predição
            prediction = self.model.predict(img_array)[0][0]
            
            # Atualiza a UI com o resultado
            self.update_results(prediction)

        except Exception as e:
            self.result_label.setText("Erro na Predição")
            self.confidence_label.setText(str(e))
            self.result_label.setStyleSheet("color: red;")

    def update_results(self, prediction_value):
        """Atualiza os rótulos de resultado e confiança com base na predição."""
        if prediction_value > 0.5:
            diagnosis = "PNEUMONIA"
            confidence = prediction_value
            self.result_label.setStyleSheet("color: #d9534f;") # Vermelho
        else:
            diagnosis = "NORMAL"
            confidence = 1 - prediction_value
            self.result_label.setStyleSheet("color: #5cb85c;") # Verde

        self.result_label.setText(f"Diagnóstico: {diagnosis}")
        self.confidence_label.setText(f"Confiança: {confidence:.2%}")

# --- Ponto de Entrada da Aplicação ---
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = XRayClassifierApp()
    window.show()
    sys.exit(app.exec_())