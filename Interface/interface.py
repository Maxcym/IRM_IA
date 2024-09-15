import sys
import torch
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget, QHBoxLayout, QSlider, QDesktopWidget, QLineEdit, QFormLayout
)
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt
from PIL import Image
from torchvision import transforms
import numpy as np
from Neural_networks.denseNet_detection import CustomDenseNet
from Neural_networks.uNet_correction_residus import UNet


class ArtifactCorrectionApp(QMainWindow):
    """
    Application pour détecter et corriger les artefacts de mouvement dans les images IRM.

    Attributs :
        image_path (str) : Chemin de l'image chargée avec des artefacts de mouvement.
        corrected_image (ndarray) : Image corrigée des artefacts de mouvement.
        predicted_residuals (ndarray) : Résidus prédits par le modèle de correction.
        current_width (int) : Largeur actuelle des images affichées après redimensionnement.
        current_height (int) : Hauteur actuelle des images affichées après redimensionnement.
    """

    def __init__(self):
        """
        Initialise l'interface utilisateur, les layouts, les widgets et les connexions des boutons.
        """
        super().__init__()
        self.setWindowTitle("Détecteur et correcteur d'artefacts de mouvement")
        self.setGeometry(600, 600, 800, 100)
        self.center()  # Centrer la fenêtre sur l'écran au lancement

        # Définir l'icône de l'application
        self.setWindowIcon(QIcon("C:/Users/maxim/PycharmProjects/artefact_detection/Demo/logo.png"))  # Remplacez par le chemin de votre logo

        # Initialisation des layouts et des widgets
        self.layout = QVBoxLayout()
        self.image_layout = QHBoxLayout()  # Layout pour afficher les images
        self.slider_layout = QHBoxLayout()  # Layout pour le curseur et sa valeur

        # Widget de conteneur pour les champs de redimensionnement
        self.resize_widget = QWidget()  # Widget conteneur pour les champs de redimensionnement et le bouton
        self.resize_layout = QFormLayout(self.resize_widget)  # Layout pour les champs de redimensionnement

        self.load_image_button = QPushButton("Charger une image avec artefacts de mouvement")
        self.detect_button = QPushButton("Détecter et corriger les artefacts de mouvement")
        self.label = QLabel("Chargez une image pour commencer !")
        self.probability_label = QLabel("Probabilité de détection : N/A")  # Label pour afficher la probabilité
        self.probability_label.setVisible(False)  # Masquer le label de probabilité au démarrage
        self.original_image_label = QLabel()  # QLabel pour l'image originale
        self.corrected_image_label = QLabel()  # QLabel pour l'image corrigée
        self.slider = QSlider(Qt.Horizontal)
        self.slider_value_label = QLabel("Facteur de correction : 1.0")  # Label pour afficher la valeur actuelle du curseur

        # Champs de saisie pour redimensionner les images
        self.width_input = QLineEdit()
        self.height_input = QLineEdit()
        self.resize_button = QPushButton("Redimensionner les images")
        self.width_input.clear()
        self.height_input.clear()
        self.resize_widget.setVisible(False)  # Masquer le widget contenant les champs de redimensionnement et le bouton

        # Masquer initialement les labels d'image et le curseur
        self.original_image_label.setVisible(False)
        self.corrected_image_label.setVisible(False)
        self.slider.setVisible(False)
        self.slider_value_label.setVisible(False)
        self.slider.setRange(1, 15)  # Plage du curseur pour ajuster le facteur de correction
        self.slider.setValue(10)  # Valeur par défaut réglée à 10 (facteur de correction de 1.0)
        self.width_input.setPlaceholderText("Largeur")
        self.height_input.setPlaceholderText("Hauteur")

        # Ajouter les widgets au layout
        self.layout.addWidget(self.load_image_button)
        self.layout.addWidget(self.detect_button)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.probability_label)
        self.slider_layout.addWidget(self.slider)
        self.slider_layout.addWidget(self.slider_value_label)
        self.layout.addLayout(self.slider_layout)
        self.image_layout.addWidget(self.original_image_label)
        self.image_layout.addWidget(self.corrected_image_label)
        self.layout.addLayout(self.image_layout)
        self.resize_layout.addRow("Largeur :", self.width_input)
        self.resize_layout.addRow("Hauteur :", self.height_input)
        self.resize_layout.addWidget(self.resize_button)
        self.layout.addWidget(self.resize_widget)

        # Définir le layout sur le widget central
        central_widget = QWidget()
        central_widget.setLayout(self.layout)
        self.setCentralWidget(central_widget)

        # Connecter les boutons aux fonctions
        self.load_image_button.clicked.connect(self.load_image_with_artifacts)
        self.detect_button.clicked.connect(self.evaluate_model)
        self.slider.valueChanged.connect(self.update_correction)
        self.resize_button.clicked.connect(self.apply_resize)

        # Variables pour stocker les chemins et le facteur de correction
        self.image_path = None
        self.corrected_image = None
        self.predicted_residuals = None
        self.current_width = None
        self.current_height = None

    def center(self):
        """Centre la fenêtre sur l'écran."""
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def load_image_with_artifacts(self):
        """
        Charge une image avec des artefacts de mouvement à partir d'un fichier.
        Réinitialise les widgets d'affichage et masque les champs inutilisés.
        """
        file_name, _ = QFileDialog.getOpenFileName(self, "Sélectionner une image avec artefacts de mouvement")
        if file_name:
            self.image_path = file_name
            self.label.setText(f"Image chargée : {file_name}")

            # Réinitialiser la visibilité des labels d'image, du curseur et des champs de saisie
            self.original_image_label.setVisible(False)
            self.corrected_image_label.setVisible(False)
            self.slider.setVisible(False)
            self.slider_value_label.setVisible(False)
            self.resize_widget.setVisible(False)
            self.width_input.clear()
            self.height_input.clear()
            self.probability_label.setVisible(False)
            self.adjustSize()

    def evaluate_model(self):
        """
        Évalue le modèle de détection des artefacts sur l'image chargée.
        Affiche la probabilité de détection et corrige les artefacts si détectés.
        """
        if not self.image_path:
            self.label.setText("Veuillez charger une image avant la détection.")
            return

        try:
            # Détection de l'appareil (CPU ou GPU)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Chargement du modèle de détection
            model = CustomDenseNet(num_classes=2, learning_rate=3e-6)
            model.load_state_dict(
                torch.load("C:/Users/maxim/PycharmProjects/artefact_detection/Neur_network/simplified_densenet_model_opti.pth", map_location=device, weights_only=True))
            model.to(device)
            model.eval()

            # Transformation de l'image
            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ])

            # Chargement et transformation de l'image de test
            image = Image.open(self.image_path)
            image = transform(image).unsqueeze(0).to(device)

            # Détection des artefacts
            output = model(image)
            probability = torch.softmax(output, dim=1)
            prob_class_1 = probability[0, 1].item() * 100  # Probabilité de la classe 1 en pourcentage
            _, predicted = torch.max(output, 1)

            # Affichage de la probabilité
            self.probability_label.setText(f"Probabilité de détection : {prob_class_1:.2f}%")
            self.probability_label.setVisible(True)

            if predicted.item() == 1:
                self.label.setText("Artefact de mouvement détecté et corrigé")
                self.correct_artifact()
            else:
                self.label.setText("Aucun artefact de mouvement détecté")
        except Exception as e:
            self.label.setText(f"Erreur lors de la détection : {e}")
            print(f"Erreur lors de l'évaluation du modèle : {e}")

    def correct_artifact(self):
        """
        Corrige les artefacts de mouvement détectés dans l'image à l'aide d'un modèle U-Net.
        Affiche les images corrigées et permet l'ajustement du facteur de correction.
        """
        try:
            # Chargement du modèle de correction
            model = UNet(learning_rate=0.00035121036825643817)
            state_dict = torch.load("C:/Users/maxim/PycharmProjects/artefact_detection/Neur_network/unet_model_residus_mouvements_opti.pth", weights_only=True)
            model.load_state_dict(state_dict, strict=False)
            model.eval()

            # Définir la transformation
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])

            # Chargement et transformation de l'image de test
            test_image = Image.open(self.image_path).convert('L')
            test_image = transform(test_image).unsqueeze(0)

            # Prédiction des résidus et stockage pour ajustement
            with torch.no_grad():
                self.predicted_residuals = model(test_image)

            # Correction initiale avec le facteur par défaut
            self.update_correction()

            # Rendre visibles le curseur, les champs de redimensionnement et le bouton
            self.slider.setVisible(True)
            self.slider_value_label.setVisible(True)
            self.resize_widget.setVisible(True)
            self.width_input.clear()
            self.height_input.clear()
            self.adjustSize()
        except Exception as e:
            self.label.setText(f"Erreur lors de la correction : {e}")
            print(f"Erreur lors de la correction des artefacts : {e}")

    def update_correction(self):
        """
        Met à jour l'image corrigée en fonction du facteur de correction actuel.
        """
        if self.predicted_residuals is not None:
            # Récupère le facteur de correction actuel à partir du curseur
            factor = self.slider.value() / 10.0
            self.slider_value_label.setText(f"Facteur de correction : {factor:.1f}")
            test_image = Image.open(self.image_path).convert('L')
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            test_image = transform(test_image).unsqueeze(0)

            # Applique le facteur de correction
            self.corrected_image = (
                test_image.squeeze().cpu().detach().numpy() -
                factor * self.predicted_residuals.squeeze().cpu().detach().numpy()
            )

            # Afficher les images sur l'interface avec les dimensions stockées
            self.display_images(test_image.squeeze().cpu().detach().numpy(), self.corrected_image, resized=True)

    def display_images(self, input_image, corrected_image, resized=False):
        """
        Affiche les images originales et corrigées sur l'interface.

        Paramètres :
            input_image (ndarray) : Image d'entrée en format numpy.
            corrected_image (ndarray) : Image corrigée en format numpy.
            resized (bool) : Indique si les images doivent être redimensionnées.
        """
        original_qpixmap = self.numpy_to_qpixmap(input_image)
        corrected_qpixmap = self.numpy_to_qpixmap(corrected_image)

        # Appliquer les dimensions stockées si les images ont été redimensionnées
        if resized and self.current_width and self.current_height:
            original_qpixmap = original_qpixmap.scaled(self.current_width, self.current_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            corrected_qpixmap = corrected_qpixmap.scaled(self.current_width, self.current_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # Définir les QPixmaps sur les labels et les rendre visibles
        self.original_image_label.setPixmap(original_qpixmap)
        self.corrected_image_label.setPixmap(corrected_qpixmap)
        self.original_image_label.setVisible(True)
        self.corrected_image_label.setVisible(True)

    def numpy_to_qpixmap(self, image_array):
        """
        Convertit un tableau numpy en QPixmap pour l'affichage.

        Paramètres :
            image_array (ndarray) : Tableau numpy représentant l'image.

        Retourne :
            QPixmap : Image convertie en format QPixmap.
        """
        # Normaliser l'image à la plage 0-255 et convertir en uint8
        image_array = (255 * (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))).astype(np.uint8)
        height, width = image_array.shape

        # Convertir le tableau numpy en QImage
        q_image = QImage(image_array.data, width, height, QImage.Format_Grayscale8)

        # Convertir QImage en QPixmap
        return QPixmap.fromImage(q_image)

    def apply_resize(self):
        """
        Redimensionne les images affichées en fonction des dimensions saisies par l'utilisateur.
        """
        try:
            width = int(self.width_input.text())
            height = int(self.height_input.text())
            self.resize_images(width, height)
        except ValueError:
            self.label.setText("Veuillez entrer des nombres valides pour la largeur et la hauteur.")

    def resize_images(self, width, height):
        """
        Redimensionne les images affichées à la largeur et hauteur spécifiées.

        Paramètres :
            width (int) : Largeur souhaitée pour les images.
            height (int) : Hauteur souhaitée pour les images.
        """
        if self.original_image_label.pixmap() and self.corrected_image_label.pixmap():
            # Stocker les dimensions redimensionnées actuelles
            self.current_width = width
            self.current_height = height

            # Redimensionner l'image originale
            resized_original = self.original_image_label.pixmap().scaled(
                width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.original_image_label.setPixmap(resized_original)

            # Redimensionner l'image corrigée
            resized_corrected = self.corrected_image_label.pixmap().scaled(
                width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.corrected_image_label.setPixmap(resized_corrected)


def main():
    """Point d'entrée principal de l'application."""
    app = QApplication(sys.argv)
    window = ArtifactCorrectionApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
