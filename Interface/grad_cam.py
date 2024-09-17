# gradcam_utils.py

import torch
import numpy as np
import cv2
from PyQt5.QtWidgets import QVBoxLayout, QLabel, QDialog
from PyQt5.QtGui import QPixmap, QImage
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import cm


class GradCamWindow(QDialog):
    """
    Fenêtre pour afficher la heatmap Grad-CAM avec une légende aux couleurs vives.

    Cette classe crée une fenêtre PyQt5 qui affiche une heatmap Grad-CAM superposée
    à l'image d'origine avec une légende de colorbar pour indiquer l'importance des régions.
    """
    def __init__(self, heatmap_mask, heatmap_image):
        """
        Initialise la fenêtre Grad-CAM.

        :param heatmap_mask: Masque de la heatmap Grad-CAM redimensionné.
        :param heatmap_image: Image d'origine sur laquelle la heatmap est superposée.
        """
        super().__init__()
        self.setWindowTitle("Grad-CAM Heatmap")
        self.setGeometry(100, 100, 500, 500)  # Taille de la fenêtre

        # Redimensionner la heatmap pour correspondre à l'image d'origine
        heatmap_resized = self.resize_heatmap_to_image(heatmap_mask, heatmap_image)

        # Créer l'image de la heatmap avec la légende
        pixmap = self.create_heatmap_with_legend(heatmap_resized, heatmap_image)

        # Configurer l'affichage de l'image
        self.image_label = QLabel(self)
        self.image_label.setPixmap(pixmap)

        # Mettre l'image dans un layout
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        self.setLayout(layout)

    def resize_heatmap_to_image(self, heatmap_mask, heatmap_image):
        """
        Redimensionne le masque de la heatmap pour correspondre exactement à l'image d'origine.

        :param heatmap_mask: Masque de la heatmap Grad-CAM.
        :param heatmap_image: Image d'origine utilisée comme référence pour le redimensionnement.
        :return: Masque de la heatmap redimensionné aux dimensions de l'image d'origine.
        """
        heatmap_mask_np = np.array(heatmap_mask)
        heatmap_resized = cv2.resize(heatmap_mask_np, (heatmap_image.width, heatmap_image.height), interpolation=cv2.INTER_LINEAR)
        return heatmap_resized

    def create_heatmap_with_legend(self, heatmap_mask, heatmap_image):
        """
        Crée une heatmap superposée à l'image d'origine avec une légende.

        :param heatmap_mask: Masque de la heatmap Grad-CAM redimensionné.
        :param heatmap_image: Image d'origine sur laquelle la heatmap est superposée.
        :return: QPixmap contenant l'image avec la heatmap et la légende.
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        im_background = ax.imshow(heatmap_image)
        im_heatmap = ax.imshow(heatmap_mask, cmap='jet', alpha=0.2)
        cbar = plt.colorbar(im_heatmap, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Importance de la décision')
        ax.axis('off')

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        image = Image.frombuffer('RGBA', canvas.get_width_height(), buf, 'raw', 'RGBA', 0, 1)
        plt.close(fig)

        qimage = QImage(image.tobytes(), image.width, image.height, QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qimage)
        return pixmap


class GradCAM:
    """
    Classe pour générer des heatmaps Grad-CAM à partir d'un modèle PyTorch.

    Cette classe capture les activations et les gradients de la couche cible
    d'un modèle de réseau de neurones pour générer une heatmap Grad-CAM.
    """
    def __init__(self, model, target_layer):
        """
        Initialise la Grad-CAM avec le modèle et la couche cible.

        :param model: Modèle PyTorch à partir duquel la Grad-CAM sera générée.
        :param target_layer: Couche cible du modèle où les activations seront extraites.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        self.hook_layers()

    def hook_layers(self):
        """
        Configure les hooks pour capturer les activations et les gradients
        de la couche cible lors de l'inférence du modèle.
        """
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        hook_forward = self.target_layer.register_forward_hook(forward_hook)
        hook_backward = self.target_layer.register_full_backward_hook(backward_hook)
        self.hooks.append(hook_forward)
        self.hooks.append(hook_backward)

    def generate_cam(self, input_image, class_index=None):
        """
        Génère la heatmap Grad-CAM pour une image d'entrée donnée.

        :param input_image: Image d'entrée sous forme de tenseur.
        :param class_index: Index de la classe pour laquelle la heatmap est générée.
                            Si None, la classe avec la plus haute probabilité est choisie.
        :return: Heatmap Grad-CAM sous forme de tableau numpy.
        """
        self.model.zero_grad()
        output = self.model(input_image)

        if class_index is None:
            class_index = torch.argmax(output)

        output[:, class_index].backward()

        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_image.shape[2], input_image.shape[3]))
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
        return cam

    def remove_hooks(self):
        """
        Supprime les hooks enregistrés sur la couche cible pour éviter
        les effets de bord lors de l'inférence ultérieure du modèle.
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def visualize_cam_on_image(image, mask):
    """
    Superpose la heatmap Grad-CAM sur l'image d'origine pour la visualisation.

    :param image: Image d'origine sous forme de tableau numpy ou PIL.
    :param mask: Heatmap Grad-CAM générée sous forme de tableau numpy.
    :return: Image PIL avec la heatmap superposée et masque normalisé.
    """
    image = np.array(image)

    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[2] == 4:
        image = image[..., :3]

    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    if mask.ndim == 2:
        mask = np.expand_dims(mask, axis=2)
    mask = np.repeat(mask, 3, axis=2)

    heatmap = cm.jet(mask[..., 0])[..., :3]
    heatmap = np.uint8(255 * heatmap)

    if heatmap.shape[2] == 4:
        heatmap = heatmap[..., :3]

    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    superimposed_img = heatmap * 0.4 + image * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    return Image.fromarray(superimposed_img), mask[..., 0]
