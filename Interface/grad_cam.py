import torch
import numpy as np
import cv2
from matplotlib import cm
from PIL import Image

class GradCAM:
    """
    Classe pour générer des heatmaps Grad-CAM à partir d'un modèle de réseau de neurones.

    Attributs:
        model (torch.nn.Module) : Le modèle pour lequel générer les heatmaps.
        target_layer (torch.nn.Module) : La couche cible à partir de laquelle extraire les gradients et les activations.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        """Configure les hooks pour capturer les activations et gradients de la couche cible."""
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_image, class_index=None):
        """
        Génère une heatmap Grad-CAM pour l'image d'entrée.

        Paramètres:
            input_image (torch.Tensor) : L'image d'entrée pour laquelle générer la heatmap.
            class_index (int) : L'indice de la classe pour laquelle calculer la heatmap.

        Retourne:
            np.ndarray : La heatmap Grad-CAM normalisée.
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

def visualize_cam_on_image(image, mask):
    """
    Superpose la heatmap Grad-CAM sur l'image d'origine.

    Paramètres:
        image (PIL.Image) : L'image d'origine.
        mask (np.ndarray) : La heatmap Grad-CAM.

    Retourne:
        PIL.Image, np.ndarray : L'image superposée et le mask pour la légende.
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
