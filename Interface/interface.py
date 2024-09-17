import sys
import torch
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget, QHBoxLayout, QSlider,
    QDesktopWidget, QLineEdit, QFormLayout
)
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt
from PIL import Image
from torchvision import transforms
import numpy as np
from Neural_networks.denseNet_detection import CustomDenseNet
from Neural_networks.uNet_correction_residus import UNet
from grad_cam import GradCamWindow, GradCAM, visualize_cam_on_image  # Importer les classes et fonctions nécessaires

class ArtifactCorrectionApp(QMainWindow):
    """
    Application pour détecter et corriger les artefacts de mouvement dans les images.

    Cette classe gère l'interface utilisateur pour charger des images, détecter les artefacts
    de mouvement, appliquer des corrections et afficher des visualisations Grad-CAM.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Motion Artifact Detector and Corrector")
        self.setGeometry(600, 600, 800, 100)
        self.center()

        # Définir l'icône de l'application
        self.setWindowIcon(QIcon("C:/Users/maxim/PycharmProjects/IRM_IA/Interface/logo.png"))

        # Initialiser les layouts et les widgets
        self.layout = QVBoxLayout()
        self.image_layout = QHBoxLayout()
        self.slider_layout = QHBoxLayout()

        # Conteneur pour les champs de redimensionnement
        self.resize_widget = QWidget()
        self.resize_layout = QFormLayout(self.resize_widget)

        # Boutons et labels
        self.load_image_button = QPushButton("Load Image with Motion Artifacts")
        self.load_image_button.setStyleSheet(
            "background-color: grey; color: white; font-weight: bold; font-size: 13px; font-family: Arial; border-radius: 3px; padding: 5px")
        self.detect_button = QPushButton("Detect and Correct Motion Artifacts")
        self.detect_button.setStyleSheet(
            "background-color: grey; color: white; font-weight: bold; font-size: 13px; font-family: Arial; border-radius: 3px; padding: 5px")
        self.show_gradcam_button = QPushButton("Show Grad-CAM")
        self.show_gradcam_button.setStyleSheet(
            "background-color: grey; color: white; font-weight: bold; font-size: 13px; font-family: Arial; border-radius: 3px; padding: 5px")
        self.label = QLabel("Load an image to start!")
        self.label.setStyleSheet("color: black; font-weight: bold")

        self.probability_label = QLabel("Detection Probability: N/A")
        self.probability_label.setStyleSheet("color: red; font-weight: bold")

        self.original_image_label = QLabel()
        self.corrected_image_label = QLabel()
        self.slider = QSlider(Qt.Horizontal)
        self.slider_value_label = QLabel("Correction Factor: 1.0")

        # Champs de redimensionnement
        self.width_input = QLineEdit()
        self.height_input = QLineEdit()
        self.resize_button = QPushButton("Resize Images")
        self.resize_button.setStyleSheet(
            "background-color: grey; color: white; font-weight: bold; font-size: 13px; font-family: Arial; border-radius: 3px; padding: 5px")

        # Définir la visibilité initiale des widgets
        self.probability_label.setVisible(False)
        self.original_image_label.setVisible(False)
        self.corrected_image_label.setVisible(False)
        self.slider.setVisible(False)
        self.slider_value_label.setVisible(False)
        self.resize_widget.setVisible(False)
        self.width_input.clear()
        self.height_input.clear()
        self.slider.setRange(1, 15)
        self.slider.setValue(10)
        self.width_input.setPlaceholderText("Width")
        self.height_input.setPlaceholderText("Height")

        self.show_gradcam_button.setVisible(False)

        # Ajouter les widgets au layout
        self.layout.addWidget(self.load_image_button)
        self.layout.addWidget(self.detect_button)
        self.layout.addWidget(self.show_gradcam_button)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.probability_label)
        self.slider_layout.addWidget(self.slider)
        self.slider_layout.addWidget(self.slider_value_label)
        self.layout.addLayout(self.slider_layout)
        self.image_layout.addWidget(self.original_image_label)
        self.image_layout.addWidget(self.corrected_image_label)
        self.layout.addLayout(self.image_layout)
        self.resize_layout.addRow("Width:", self.width_input)
        self.resize_layout.addRow("Height:", self.height_input)
        self.resize_layout.addWidget(self.resize_button)
        self.layout.addWidget(self.resize_widget)

        # Définir le layout du widget central
        central_widget = QWidget()
        central_widget.setLayout(self.layout)
        self.setCentralWidget(central_widget)

        # Connecter les boutons aux fonctions
        self.load_image_button.clicked.connect(self.load_image_with_artifacts)
        self.detect_button.clicked.connect(self.evaluate_model)
        self.show_gradcam_button.clicked.connect(self.show_gradcam)
        self.slider.valueChanged.connect(self.update_correction)
        self.resize_button.clicked.connect(self.apply_resize)

        # Variables pour stocker les chemins et le facteur de correction
        self.image_path = None
        self.corrected_image = None
        self.predicted_residuals = None
        self.current_width = None
        self.current_height = None

    def center(self):
        """
        Centre la fenêtre sur l'écran.
        """
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def load_image_with_artifacts(self):
        """
        Charge une image contenant des artefacts de mouvement.
        """
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image with Motion Artifacts")
        if file_name:
            self.image_path = file_name
            self.label.setText(f"Loaded image: {file_name}")
            self.reset_interface()

    def reset_interface(self):
        """
        Réinitialise l'interface après le chargement d'une nouvelle image.
        """
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
        Évalue l'image chargée avec le modèle de détection d'artefacts.
        """
        if not self.image_path:
            self.label.setText("Please load an image before detection.")
            return

        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model = CustomDenseNet(num_classes=2, learning_rate=3e-6)
            model.load_state_dict(
                torch.load("C:/Users/maxim/PycharmProjects/IRM_IA/Neural_networks/simplified_densenet_model_opti.pth", map_location=device, weights_only=True))
            model.to(device)
            model.eval()

            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ])

            image = Image.open(self.image_path)
            image = transform(image).unsqueeze(0).to(device)

            output = model(image)
            probability = torch.softmax(output, dim=1)
            prob_class_1 = probability[0, 1].item() * 100
            _, predicted = torch.max(output, 1)

            self.probability_label.setText(f"Detection Probability: {prob_class_1:.2f}%")
            self.probability_label.setVisible(True)

            if predicted.item() == 1:
                self.label.setText("Motion artifact detected and corrected")
                self.correct_artifact()
            else:
                self.label.setText("No motion artifacts detected")
        except Exception as e:
            self.label.setText(f"Error during detection: {e}")

    def correct_artifact(self):
        """
        Corrige les artefacts de mouvement détectés dans l'image chargée.
        """
        try:
            model = UNet(learning_rate=0.00035121036825643817)
            state_dict = torch.load(
                "C:/Users/maxim/PycharmProjects/IRM_IA/Neural_networks/unet_model_residus_mouvements_opti.pth",
                weights_only=True)
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
            self.resize_widget.setVisible(True)
            self.width_input.clear()
            self.height_input.clear()
            self.adjustSize()
        except Exception as e:
            self.label.setText(f"Error during correction: {e}")

    def update_correction(self):
        """
        Met à jour l'image corrigée en fonction du facteur de correction sélectionné.
        """
        if self.predicted_residuals is not None:
            factor = self.slider.value() / 10.0
            self.slider_value_label.setText(f"Correction Factor: {factor:.1f}")

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

            self.display_images(test_image.squeeze().cpu().detach().numpy(), self.corrected_image, resized=True)

    def display_images(self, input_image, corrected_image, resized=False):
        """
        Affiche l'image originale et l'image corrigée dans l'interface.

        :param input_image: Image originale sous forme de tableau numpy.
        :param corrected_image: Image corrigée sous forme de tableau numpy.
        :param resized: Booléen indiquant si les images doivent être redimensionnées.
        """
        original_qpixmap = self.numpy_to_qpixmap(input_image)
        corrected_qpixmap = self.numpy_to_qpixmap(corrected_image)

        if resized and self.current_width and self.current_height:
            original_qpixmap = original_qpixmap.scaled(self.current_width, self.current_height, Qt.KeepAspectRatio,
                                                       Qt.SmoothTransformation)
            corrected_qpixmap = corrected_qpixmap.scaled(self.current_width, self.current_height, Qt.KeepAspectRatio,
                                                         Qt.SmoothTransformation)

        self.original_image_label.setPixmap(original_qpixmap)
        self.corrected_image_label.setPixmap(corrected_qpixmap)
        self.original_image_label.setVisible(True)
        self.corrected_image_label.setVisible(True)
        self.show_gradcam_button.setVisible(True)

    def numpy_to_qpixmap(self, image_array):
        """
        Convertit un tableau numpy en QPixmap pour l'affichage dans l'interface.

        :param image_array: Tableau numpy représentant l'image à convertir.
        :return: QPixmap correspondant à l'image.
        """
        image_array = (255 * (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))).astype(
            np.uint8)
        height, width = image_array.shape
        q_image = QImage(image_array.data, width, height, QImage.Format_Grayscale8)
        return QPixmap.fromImage(q_image)

    def apply_resize(self):
        """
        Applique le redimensionnement des images affichées en fonction des dimensions saisies par l'utilisateur.
        """
        try:
            width = int(self.width_input.text())
            height = int(self.height_input.text())
            self.resize_images(width, height)
        except ValueError:
            self.label.setText("Please enter valid numbers for width and height.")

    def resize_images(self, width, height):
        """
        Redimensionne les images affichées selon les dimensions spécifiées.

        :param width: Largeur souhaitée.
        :param height: Hauteur souhaitée.
        """
        if self.original_image_label.pixmap() and self.corrected_image_label.pixmap():
            self.current_width = width
            self.current_height = height

            resized_original = self.original_image_label.pixmap().scaled(
                width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.original_image_label.setPixmap(resized_original)

            resized_corrected = self.corrected_image_label.pixmap().scaled(
                width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.corrected_image_label.setPixmap(resized_corrected)

    def show_gradcam(self):
        """
        Affiche la visualisation Grad-CAM dans une fenêtre secondaire avec légende.
        """
        if not self.image_path:
            self.label.setText("Veuillez charger une image avant d'afficher Grad-CAM.")
            return

        try:
            # Détecter le dispositif (CPU ou GPU)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Charger le modèle de détection
            model = CustomDenseNet(num_classes=2, learning_rate=3e-6)
            model.load_state_dict(
                torch.load("C:/Users/maxim/PycharmProjects/IRM_IA/Neural_networks/simplified_densenet_model_opti.pth",
                           map_location=device, weights_only=True))
            model.to(device)
            model.eval()

            # Transformer l'image
            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ])

            # Charger et transformer l'image de test
            image = Image.open(self.image_path).convert('L')
            input_image = transform(image).unsqueeze(0).to(device)

            # Sélectionner la couche cible pour Grad-CAM
            target_layer = model.blocks[-1].denseblock[-1].conv
            grad_cam = GradCAM(model, target_layer)

            # Générer le masque Grad-CAM
            cam_mask = grad_cam.generate_cam(input_image)
            grad_cam.remove_hooks()  # Nettoyer les hooks après utilisation

            # Visualiser la heatmap sur l'image originale
            heatmap_image, heatmap_mask = visualize_cam_on_image(image, cam_mask)

            # Afficher la heatmap avec la légende dans une petite fenêtre
            self.gradcam_window = GradCamWindow(heatmap_mask, heatmap_image)
            self.gradcam_window.exec_()

        except Exception as e:
            self.label.setText(f"Erreur lors de l'affichage de Grad-CAM: {e}")
            print(f"Erreur pendant la visualisation de Grad-CAM: {e}")


def main():
    """
    Point d'entrée principal de l'application.
    """
    app = QApplication(sys.argv)
    window = ArtifactCorrectionApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
