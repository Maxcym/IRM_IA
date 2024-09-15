import sys
import torch
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget, QHBoxLayout, QSlider, QDesktopWidget
)
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt
from PIL import Image
from torchvision import transforms
import numpy as np
from Neural_networks.denseNet_detection import CustomDenseNet
from Neural_networks.uNet_correction_residus import UNet
from grad_cam import GradCAM, visualize_cam_on_image  # Import Grad-CAM related code
import cv2

class ArtifactCorrectionApp(QMainWindow):
    """
    Application pour détecter et corriger les artefacts de mouvement dans les images IRM.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Détecteur et correcteur d'artefacts de mouvement")
        self.setGeometry(600, 600, 800, 100)
        self.center()

        # Définir l'icône de l'application
        self.setWindowIcon(QIcon("C:/Users/maxim/PycharmProjects/artefact_detection/Demo/logo.png"))

        # Initialisation des layouts et des widgets
        self.layout = QVBoxLayout()
        self.image_layout = QHBoxLayout()
        self.slider_layout = QHBoxLayout()

        self.toggle_grad_cam_button = QPushButton("Afficher/Masquer la Grad-CAM")
        self.toggle_grad_cam_button.setStyleSheet("background-color: grey; color: white; font-weight: bold; font-size: 13px; font-family: Arial; border-radius: 3px; padding: 5px")
        self.toggle_grad_cam_button.setVisible(False)  # Initially hidden
        self.grad_cam_label = QLabel()
        self.load_image_button = QPushButton("Load Image with motion artifacts")
        self.load_image_button.setStyleSheet("background-color: grey; color: white; font-weight: bold; font-size: 13px; font-family: Arial; border-radius: 3px; padding: 5px")
        self.detect_button = QPushButton("Detect and correct motion artifacts")
        self.detect_button.setStyleSheet("background-color: grey; color: white; font-weight: bold; font-size: 13px; font-family: Arial; border-radius: 3px; padding: 5px")
        self.label = QLabel("Load an image to start!")
        self.label.setStyleSheet("color: black; font-weight: bold; font-size: 11px; font-family: Arial")
        self.probability_label = QLabel("Detection probability: N/A")  # Label to display probability
        self.probability_label.setStyleSheet("color: red; font-weight: bold; font-size: 11px; font-family: Arial")
        self.probability_label.setVisible(False)  # Hide the probability label initially
        self.original_image_label = QLabel()  # QLabel for original image
        self.corrected_image_label = QLabel()  # QLabel for corrected image
        self.slider = QSlider(Qt.Horizontal)
        self.slider_value_label = QLabel("Correction factor: 1.0")  # Label to display current slider value
        self.slider_value_label.setStyleSheet("font-family: Arial")

        # Masquer initialement les labels d'image et le curseur
        self.original_image_label.setVisible(False)
        self.corrected_image_label.setVisible(False)
        self.grad_cam_label.setVisible(False)
        self.slider.setVisible(False)
        self.slider_value_label.setVisible(False)
        self.slider.setRange(1, 15)
        self.slider.setValue(10)

        # Ajouter les widgets au layout
        self.layout.addWidget(self.load_image_button)
        self.layout.addWidget(self.detect_button)
        self.layout.addWidget(self.toggle_grad_cam_button)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.probability_label)
        self.slider_layout.addWidget(self.slider)
        self.slider_layout.addWidget(self.slider_value_label)
        self.layout.addLayout(self.slider_layout)
        self.image_layout.addWidget(self.original_image_label)
        self.image_layout.addWidget(self.corrected_image_label)
        self.image_layout.addWidget(self.grad_cam_label)
        self.layout.addLayout(self.image_layout)

        # Définir le layout sur le widget central
        central_widget = QWidget()
        central_widget.setLayout(self.layout)
        self.setCentralWidget(central_widget)

        # Connecter les boutons aux fonctions
        self.load_image_button.clicked.connect(self.load_image_with_artifacts)
        self.detect_button.clicked.connect(self.evaluate_model)
        self.toggle_grad_cam_button.clicked.connect(self.toggle_grad_cam)
        self.slider.valueChanged.connect(self.update_correction)

        # Variables pour stocker les chemins et le facteur de correction
        self.image_path = None
        self.corrected_image = None
        self.predicted_residuals = None
        self.current_width = None
        self.current_height = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.target_layer = None

    def center(self):
        """Centre la fenêtre sur l'écran."""
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def load_image_with_artifacts(self):
        """Charge une image avec des artefacts de mouvement à partir d'un fichier."""
        file_name, _ = QFileDialog.getOpenFileName(self, "Sélectionner une image avec artefacts de mouvement")
        if file_name:
            self.image_path = file_name
            self.label.setText(f"Image chargée : {file_name}")

            # Réinitialiser la visibilité des labels d'image, du curseur et des champs de saisie
            self.original_image_label.setVisible(False)
            self.corrected_image_label.setVisible(False)
            self.grad_cam_label.setVisible(False)
            self.slider.setVisible(False)
            self.slider_value_label.setVisible(False)
            self.probability_label.setVisible(False)
            self.toggle_grad_cam_button.setVisible(False)
            self.adjustSize()

    def evaluate_model(self):
        """Évalue le modèle de détection des artefacts sur l'image chargée."""
        if not self.image_path:
            self.label.setText("Veuillez charger une image avant la détection.")
            return

        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Chargement du modèle de détection
            self.model = CustomDenseNet(num_classes=2, learning_rate=3e-6)
            self.model.load_state_dict(
                torch.load("C:/Users/maxim/PycharmProjects/artefact_detection/Neur_network/simplified_densenet_model_opti.pth", map_location=self.device, weights_only=True))
            self.model.to(self.device)
            self.model.eval()

            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ])

            image = Image.open(self.image_path)
            image = transform(image).unsqueeze(0).to(self.device)

            output = self.model(image)
            probability = torch.softmax(output, dim=1)
            prob_class_1 = probability[0, 1].item() * 100
            _, predicted = torch.max(output, 1)

            self.probability_label.setText(f"Probabilité de détection : {prob_class_1:.2f}%")
            self.probability_label.setVisible(True)

            if predicted.item() == 1:
                self.label.setText("Artefact de mouvement détecté et corrigé")
                self.display_grad_cam()
                self.correct_artifact()
            else:
                self.label.setText("Aucun artefact de mouvement détecté")
        except Exception as e:
            self.label.setText(f"Erreur lors de la détection : {e}")
            print(f"Erreur lors de l'évaluation du modèle : {e}")

    def correct_artifact(self):
        """Corrige les artefacts de mouvement détectés dans l'image à l'aide d'un modèle U-Net."""
        try:
            model = UNet(learning_rate=0.00035121036825643817)
            state_dict = torch.load("C:/Users/maxim/PycharmProjects/artefact_detection/Neur_network/unet_model_residus_mouvements_opti.pth", weights_only=True)
            model.load_state_dict(state_dict, strict=False)
            model.eval()

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])

            test_image = Image.open(self.image_path).convert('L')
            test_image = transform(test_image).unsqueeze(0)

            with torch.no_grad():
                self.predicted_residuals = model(test_image)

            self.update_correction()

            self.slider.setVisible(True)
            self.slider_value_label.setVisible(True)
            self.adjustSize()
        except Exception as e:
            self.label.setText(f"Erreur lors de la correction : {e}")
            print(f"Erreur lors de la correction des artefacts : {e}")

    def update_correction(self):
        """Met à jour l'image corrigée en fonction du facteur de correction actuel."""
        if self.predicted_residuals is not None:
            factor = self.slider.value() / 10.0
            self.slider_value_label.setText(f"Facteur de correction : {factor:.1f}")
            test_image = Image.open(self.image_path).convert('L')
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            test_image = transform(test_image).unsqueeze(0)

            self.corrected_image = (
                test_image.squeeze().cpu().detach().numpy() -
                factor * self.predicted_residuals.squeeze().cpu().detach().numpy()
            )

            self.display_images(test_image.squeeze().cpu().detach().numpy(), self.corrected_image)

    def display_images(self, input_image, corrected_image):
        """Affiche les images originales et corrigées sur l'interface."""
        original_qpixmap = self.numpy_to_qpixmap(input_image)
        corrected_qpixmap = self.numpy_to_qpixmap(corrected_image)

        # Définir la taille par défaut à 350x350 pixels
        default_width = 350
        default_height = 350

        original_qpixmap = original_qpixmap.scaled(default_width, default_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        corrected_qpixmap = corrected_qpixmap.scaled(default_width, default_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        self.original_image_label.setPixmap(original_qpixmap)
        self.corrected_image_label.setPixmap(corrected_qpixmap)
        self.original_image_label.setVisible(True)
        self.corrected_image_label.setVisible(True)

    def numpy_to_qpixmap(self, image_array):
        """Convertit un tableau numpy en QPixmap pour l'affichage."""
        image_array = (255 * (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))).astype(np.uint8)
        height, width = image_array.shape
        q_image = QImage(image_array.data, width, height, QImage.Format_Grayscale8)
        return QPixmap.fromImage(q_image)

    def display_grad_cam(self):
        """Affiche la heatmap Grad-CAM superposée sur l'image chargée dans l'interface principale."""
        if self.model is None or self.image_path is None:
            self.label.setText("Erreur: Modèle ou image non chargé.")
            return

        try:
            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])

            image = Image.open(self.image_path)
            input_image = transform(image).unsqueeze(0).to(self.device)

            self.target_layer = self.model.blocks[-1].denseblock[-1].conv

            grad_cam = GradCAM(self.model, self.target_layer)

            cam_mask = grad_cam.generate_cam(input_image)

            heatmap_image, _ = visualize_cam_on_image(image, cam_mask)

            pixmap = self.pil_image_to_qpixmap(heatmap_image)

            # Redimensionner la Grad-CAM à 350x350 pixels par défaut
            default_width = 350
            default_height = 350
            pixmap = pixmap.scaled(default_width, default_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            self.grad_cam_label.setPixmap(pixmap)
            self.grad_cam_label.setVisible(False)
            self.toggle_grad_cam_button.setVisible(True)
        except Exception as e:
            self.label.setText(f"Erreur lors de l'affichage de Grad-CAM : {e}")
            print(f"Erreur lors de l'affichage de Grad-CAM : {e}")

    def toggle_grad_cam(self):
        """Bascule la visibilité de la heatmap Grad-CAM."""
        self.grad_cam_label.setVisible(not self.grad_cam_label.isVisible())
    def pil_image_to_qpixmap(self, image):
        """Convertit une image PIL en QPixmap."""
        image = image.convert("RGB")
        data = image.tobytes("raw", "RGB")
        q_image = QImage(data, image.width, image.height, QImage.Format_RGB888)
        return QPixmap.fromImage(q_image)


def main():
    """Point d'entrée principal de l'application."""
    app = QApplication(sys.argv)
    window = ArtifactCorrectionApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
