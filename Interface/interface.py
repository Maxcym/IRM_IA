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
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Motion artifact detector and corrector")
        self.setGeometry(600, 600, 800, 100)
        self.center()  # Center the window on the screen at launch

        # Set the application icon
        self.setWindowIcon(QIcon("C:/Users/maxim/PycharmProjects/IRM_IA/Interface/logo.png"))  # Replace with your logo path

        # Initialize layout and widgets
        self.layout = QVBoxLayout()
        self.image_layout = QHBoxLayout()  # Layout for displaying images
        self.slider_layout = QHBoxLayout()  # Layout for slider and its value display

        # Resize input fields container
        self.resize_widget = QWidget()  # Container widget for resizing inputs and button
        self.resize_layout = QFormLayout(self.resize_widget)  # Layout for resizing inputs

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


        # Resize input fields
        self.width_input = QLineEdit()
        self.height_input = QLineEdit()
        self.resize_button = QPushButton("Resize images")
        self.resize_button.setStyleSheet("background-color: grey; color: white; font-weight: bold; font-size: 13px; font-family: Arial; border-radius: 3px; padding: 5px")
        # Clear the input fields on startup
        self.width_input.clear()
        self.height_input.clear()
        self.resize_widget.setVisible(False)  # Hide the entire widget containing resize fields and button

        # Initially hide image labels and slider
        self.original_image_label.setVisible(False)
        self.corrected_image_label.setVisible(False)
        self.slider.setVisible(False)
        self.slider_value_label.setVisible(False)
        self.slider.setRange(1, 15)  # Slider range to adjust correction factor
        self.slider.setValue(10)  # Default value is set to 10 (correction factor of 1.0)
        self.width_input.setPlaceholderText("Width")
        self.height_input.setPlaceholderText("Height")

        # Add widgets to layout
        self.layout.addWidget(self.load_image_button)
        self.layout.addWidget(self.detect_button)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.probability_label)  # Add probability label to the layout
        self.slider_layout.addWidget(self.slider)
        self.slider_layout.addWidget(self.slider_value_label)
        self.layout.addLayout(self.slider_layout)
        self.image_layout.addWidget(self.original_image_label)
        self.image_layout.addWidget(self.corrected_image_label)
        self.layout.addLayout(self.image_layout)
        self.resize_layout.addRow("Width:", self.width_input)
        self.resize_layout.addRow("Height:", self.height_input)
        self.resize_layout.addWidget(self.resize_button)  # Add the button to the same layout
        self.layout.addWidget(self.resize_widget)  # Add the container widget to the main layout

        # Set the layout to the central widget
        central_widget = QWidget()
        central_widget.setLayout(self.layout)
        self.setCentralWidget(central_widget)

        # Connect buttons to functions
        self.load_image_button.clicked.connect(self.load_image_with_artifacts)
        self.detect_button.clicked.connect(self.evaluate_model)
        self.slider.valueChanged.connect(self.update_correction)
        self.resize_button.clicked.connect(self.apply_resize)

        # Variables to store paths and correction factor
        self.image_path = None
        self.corrected_image = None
        self.predicted_residuals = None
        self.current_width = None  # To store the current width of images
        self.current_height = None  # To store the current height of images

    def center(self):
        """Center the window on the screen."""
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def load_image_with_artifacts(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select image with motion artifacts")
        if file_name:
            self.image_path = file_name
            self.label.setText(f"Loaded image: {file_name}")

            # Reset visibility of image labels, slider, and input fields
            self.original_image_label.setVisible(False)
            self.corrected_image_label.setVisible(False)
            self.slider.setVisible(False)
            self.slider_value_label.setVisible(False)
            self.resize_widget.setVisible(False)  # Hide the entire widget containing resize fields and button
            self.width_input.clear()  # Clear the width input field
            self.height_input.clear()  # Clear the height input field
            self.probability_label.setVisible(False)  # Hide probability label when a new image is loaded
            self.adjustSize()  # Adjust window size after hiding images

    def evaluate_model(self):
        if not self.image_path:
            self.label.setText("Please load an image before detection.")
            return

        try:
            # Detect device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Load detection model
            model = CustomDenseNet(num_classes=2, learning_rate=3e-6)
            model.load_state_dict(
                torch.load("C:/Users/maxim/PycharmProjects/IRM_IA/Neural_networks/simplified_densenet_model_opti.pth", map_location=device, weights_only=True))
            model.to(device)
            model.eval()

            # Transform image
            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ])

            # Load and transform the test image
            image = Image.open(self.image_path)
            image = transform(image).unsqueeze(0).to(device)

            # Run detection
            output = model(image)
            probability = torch.softmax(output, dim=1)
            prob_class_1 = probability[0, 1].item() * 100  # Probability of class 1 in percentage
            _, predicted = torch.max(output, 1)

            # Display probability
            self.probability_label.setText(f"Detection Probability: {prob_class_1:.2f}%")
            self.probability_label.setVisible(True)  # Show the probability label after detection

            if predicted.item() == 1:
                self.label.setText("Motion artifact detected and corrected")
                self.correct_artifact()
            else:
                self.label.setText("No motion artifacts detected")
        except Exception as e:
            self.label.setText(f"Error during detection: {e}")
            print(f"Error during model evaluation: {e}")

    def correct_artifact(self):
        try:
            # Load correction model
            model = UNet(learning_rate=0.00035121036825643817)
            state_dict = torch.load("C:/Users/maxim/PycharmProjects/IRM_IA/Neural_networks/unet_model_residus_mouvements_opti.pth", weights_only=True)
            model.load_state_dict(state_dict, strict=False)
            model.eval()

            # Define transform
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])

            # Load and transform the test image
            test_image = Image.open(self.image_path).convert('L')
            test_image = transform(test_image).unsqueeze(0)

            # Predict residuals and store them for adjustment
            with torch.no_grad():
                self.predicted_residuals = model(test_image)

            # Initial correction with default factor
            self.update_correction()

            # Make the slider, resize inputs, and resize button visible
            self.slider.setVisible(True)
            self.slider_value_label.setVisible(True)
            self.resize_widget.setVisible(True)
            self.width_input.clear()
            self.height_input.clear()
            self.adjustSize()
        except Exception as e:
            self.label.setText(f"Error during correction: {e}")
            print(f"Error during artifact correction: {e}")

    def update_correction(self):
        if self.predicted_residuals is not None:
            # Get the current correction factor from the slider
            factor = self.slider.value() / 10.0  # Convert slider value to a float factor
            self.slider_value_label.setText(f"Correction Factor: {factor:.1f}")  # Update label with current factor
            test_image = Image.open(self.image_path).convert('L')
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            test_image = transform(test_image).unsqueeze(0)

            # Apply correction factor
            self.corrected_image = (
                test_image.squeeze().cpu().detach().numpy() -
                factor * self.predicted_residuals.squeeze().cpu().detach().numpy()
            )

            # Display images on the interface with stored dimensions
            self.display_images(test_image.squeeze().cpu().detach().numpy(), self.corrected_image, resized=True)

    def display_images(self, input_image, corrected_image, resized=False):
        # Convert images to QPixmap
        original_qpixmap = self.numpy_to_qpixmap(input_image)
        corrected_qpixmap = self.numpy_to_qpixmap(corrected_image)

        # If images were resized, apply the stored dimensions
        if resized and self.current_width and self.current_height:
            original_qpixmap = original_qpixmap.scaled(self.current_width, self.current_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            corrected_qpixmap = corrected_qpixmap.scaled(self.current_width, self.current_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # Set QPixmap to labels and make them visible
        self.original_image_label.setPixmap(original_qpixmap)
        self.corrected_image_label.setPixmap(corrected_qpixmap)
        self.original_image_label.setVisible(True)
        self.corrected_image_label.setVisible(True)

    def numpy_to_qpixmap(self, image_array):
        # Normalize the image to 0-255 range and convert to uint8
        image_array = (255 * (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))).astype(
            np.uint8)
        height, width = image_array.shape

        # Convert the numpy array to QImage
        q_image = QImage(image_array.data, width, height, QImage.Format_Grayscale8)

        # Convert QImage to QPixmap
        return QPixmap.fromImage(q_image)

    def apply_resize(self):
        """Resize the displayed images based on the user input for width and height."""
        try:
            width = int(self.width_input.text())
            height = int(self.height_input.text())
            self.resize_images(width, height)
        except ValueError:
            self.label.setText("Please enter valid numbers for width and height.")

    def resize_images(self, width, height):
        """Resize the displayed images to the specified width and height."""
        if self.original_image_label.pixmap() and self.corrected_image_label.pixmap():
            # Store the current resized dimensions
            self.current_width = width
            self.current_height = height

            # Resize the original image
            resized_original = self.original_image_label.pixmap().scaled(
                width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.original_image_label.setPixmap(resized_original)

            # Resize the corrected image
            resized_corrected = self.corrected_image_label.pixmap().scaled(
                width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.corrected_image_label.setPixmap(resized_corrected)


def main():
    app = QApplication(sys.argv)
    window = ArtifactCorrectionApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
