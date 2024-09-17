import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Configuration pour l'utilisation des Tensor Cores sur GPU Nvidia (si applicable)
torch.set_float32_matmul_precision('high')


class ArtefactDataset(Dataset):
    """
    Dataset personnalisé pour les images d'IRM avec et sans artefacts de mouvement.

    Attributs :
        root_dir_artefacts (str) : Chemin vers le dossier contenant les images avec artefacts.
        root_dir_targets (str) : Chemin vers le dossier contenant les images cibles sans artefacts.
        transform (callable, optionnel) : Transformations à appliquer sur les images.
        artefact_files (list) : Liste des fichiers d'images avec artefacts.
        target_files (list) : Liste des fichiers d'images cibles sans artefacts.

    Méthodes :
        __len__() : Retourne le nombre total d'images dans le dataset.
        __getitem__(idx) : Retourne un dictionnaire contenant l'image avec artefacts et le résidu de mouvement correspondant.
    """

    def __init__(self, root_dir_artefacts, root_dir_targets, transform=None):
        """
        Initialise le dataset avec les chemins des dossiers d'images et les transformations éventuelles.

        Paramètres :
            root_dir_artefacts (str) : Chemin vers le dossier contenant les images avec artefacts.
            root_dir_targets (str) : Chemin vers le dossier contenant les images sans artefacts.
            transform (callable, optionnel) : Transformations à appliquer sur les images.
        """
        self.root_dir_artefacts = root_dir_artefacts
        self.root_dir_targets = root_dir_targets
        self.transform = transform
        self.artefact_files = sorted(os.listdir(root_dir_artefacts))
        self.target_files = sorted(os.listdir(root_dir_targets))

    def __len__(self):
        """
        Retourne le nombre total d'images dans le dataset.

        Retourne :
            int : Nombre d'images dans le dataset.
        """
        return len(self.artefact_files)

    def __getitem__(self, idx):
        """
        Récupère une paire d'images (avec artefact et sans artefact) pour l'indice donné.

        Paramètres :
            idx (int) : Indice de l'image à récupérer.

        Retourne :
            dict : Dictionnaire contenant l'image avec artefacts et le résidu de mouvement calculé.
        """
        artefact_path = os.path.join(self.root_dir_artefacts, self.artefact_files[idx])
        target_path = os.path.join(self.root_dir_targets, self.target_files[idx])

        artefact_image = Image.open(artefact_path)
        target_image = Image.open(target_path)

        if self.transform:
            artefact_image = self.transform(artefact_image)
            target_image = self.transform(target_image)

        residu_mouvement = artefact_image - target_image

        return {'avec_artefact': artefact_image, 'residu_mouvement': residu_mouvement}


data_dir_artefacts = "../Dataset_Normalized/Class_1"
data_dir_targets = "../Dataset_Normalized/Class_0"

train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

val_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

full_dataset = ArtefactDataset(root_dir_artefacts=data_dir_artefacts,
                               root_dir_targets=data_dir_targets,
                               transform=train_transforms)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

val_dataset.dataset.transform = val_transforms

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)


class ConvBlock(nn.Module):
    """
    Bloc de convolution utilisé dans les encodeurs et décodeurs U-Net.

    Attributs :
        conv (nn.Sequential) : Séquence de convolutions, de BatchNorm et d'activations ReLU.

    Méthodes :
        forward(x) : Applique les convolutions et les activations sur l'entrée.
    """

    def __init__(self, in_channels, out_channels):
        """
        Initialise un bloc de convolution avec les canaux d'entrée et de sortie spécifiés.

        Paramètres :
            in_channels (int) : Nombre de canaux en entrée.
            out_channels (int) : Nombre de canaux en sortie.
        """
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Applique les convolutions et activations sur l'entrée.

        Paramètres :
            x (Tensor) : Entrée du tenseur.

        Retourne :
            Tensor : Sortie du bloc de convolution.
        """
        return self.conv(x)


class UNetEncoder(nn.Module):
    """
    Encodeur U-Net avec bloc de convolution et pooling.

    Attributs :
        conv (ConvBlock) : Bloc de convolution pour l'extraction des caractéristiques.
        pool (nn.MaxPool2d) : Couche de pooling pour réduire la taille spatiale.

    Méthodes :
        forward(x) : Applique la convolution et le pooling sur l'entrée.
    """

    def __init__(self, in_channels, out_channels):
        """
        Initialise un encodeur U-Net avec les canaux d'entrée et de sortie spécifiés.

        Paramètres :
            in_channels (int) : Nombre de canaux en entrée.
            out_channels (int) : Nombre de canaux en sortie.
        """
        super(UNetEncoder, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """
        Applique la convolution et le pooling sur l'entrée.

        Paramètres :
            x (Tensor) : Entrée du tenseur.

        Retourne :
            tuple : Sortie de la convolution et sortie après pooling.
        """
        x = self.conv(x)
        x_pooled = self.pool(x)
        return x, x_pooled


class UNetDecoder(nn.Module):
    """
    Décodeur U-Net avec déconvolution et bloc de convolution.

    Attributs :
        upconv (nn.ConvTranspose2d) : Déconvolution pour augmenter la taille spatiale.
        conv (ConvBlock) : Bloc de convolution pour fusionner les caractéristiques.

    Méthodes :
        forward(x, encoder_features) : Applique la déconv, concatène avec les features de l'encodeur, puis applique la conv.
    """

    def __init__(self, in_channels, out_channels):
        """
        Initialise un décodeur U-Net avec les canaux d'entrée et de sortie spécifiés.

        Paramètres :
            in_channels (int) : Nombre de canaux en entrée.
            out_channels (int) : Nombre de canaux en sortie.
        """
        super(UNetDecoder, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x, encoder_features):
        """
        Applique la déconv, concatène avec les features de l'encodeur, puis applique la conv.

        Paramètres :
            x (Tensor) : Entrée du décodeur.
            encoder_features (Tensor) : Caractéristiques de l'encodeur à concaténer.

        Retourne :
            Tensor : Sortie du décodeur après fusion et convolution.
        """
        x = self.upconv(x)
        x = torch.cat((x, encoder_features), dim=1)
        return self.conv(x)


class SpatialAttention(nn.Module):
    """
    Module d'attention spatiale pour les cartes de caractéristiques.

    Attributs :
        conv1 (nn.Conv2d) : Convolution pour générer la carte d'attention.
        sigmoid (nn.Sigmoid) : Fonction d'activation pour normaliser la carte d'attention.

    Méthodes :
        forward(x) : Calcule la carte d'attention et la multiplie par l'entrée.
    """

    def __init__(self, kernel_size=7):
        """
        Initialise le module d'attention spatiale.

        Paramètres :
            kernel_size (int) : Taille du noyau de convolution pour générer la carte d'attention.
        """
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Calcule la carte d'attention et la multiplie par l'entrée.

        Paramètres :
            x (Tensor) : Entrée du tenseur.

        Retourne :
            Tensor : Sortie après application de l'attention.
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class UNet(pl.LightningModule):
    """
    Modèle U-Net avec attention spatiale pour la correction des artefacts de mouvement dans les images d'IRM.

    Attributs :
        learning_rate (float) : Taux d'apprentissage pour l'entraînement.
        train_losses (list) : Liste des pertes d'entraînement pour chaque époque.
        val_losses (list) : Liste des pertes de validation pour chaque époque.
        encoder1-5 (UNetEncoder) : Encodeurs du modèle U-Net.
        decoder1-5 (UNetDecoder) : Décodeurs du modèle U-Net.
        bottleneck (ConvBlock) : Bloc de goulot d'étranglement pour la partie la plus basse du réseau.
        attention1-5 (SpatialAttention) : Modules d'attention spatiale pour les sorties des décodeurs.
        final_conv (nn.Conv2d) : Convolution finale pour produire l'image corrigée.

    Méthodes :
        forward(x) : Passe avant du modèle pour produire l'image corrigée.
        configure_optimizers() : Configure l'optimiseur et le scheduler.
        loss(outputs, target) : Calcule la perte entre les prédictions et les cibles.
        training_step(batch, batch_idx) : Effectue une étape d'entraînement et enregistre la perte.
        on_train_epoch_end() : Enregistre la perte d'entraînement à la fin de chaque époque.
        validation_step(batch, batch_idx) : Effectue une étape de validation et enregistre la perte.
        on_validation_epoch_end() : Enregistre la perte de validation à la fin de chaque époque.
        visualize_predictions(inputs, outputs, targets, batch_idx, phase) : Affiche les images pendant l'entraînement/validation.
        plot_loss() : Affiche les courbes de perte d'entraînement et de validation.
    """

    def __init__(self, learning_rate=0.00001):
        """
        Initialise le modèle U-Net avec attention spatiale et configure ses paramètres.

        Paramètres :
            learning_rate (float) : Taux d'apprentissage pour l'entraînement.
        """
        super(UNet, self).__init__()
        self.learning_rate = learning_rate
        self.train_losses = []
        self.val_losses = []

        self.encoder1 = UNetEncoder(1, 16)
        self.encoder2 = UNetEncoder(16, 32)
        self.encoder3 = UNetEncoder(32, 64)
        self.encoder4 = UNetEncoder(64, 128)
        self.encoder5 = UNetEncoder(128, 256)
        self.bottleneck = ConvBlock(256, 512)

        self.attention5 = SpatialAttention()
        self.attention4 = SpatialAttention()
        self.attention3 = SpatialAttention()
        self.attention2 = SpatialAttention()
        self.attention1 = SpatialAttention()

        self.decoder5 = UNetDecoder(512, 256)
        self.decoder4 = UNetDecoder(256, 128)
        self.decoder3 = UNetDecoder(128, 64)
        self.decoder2 = UNetDecoder(64, 32)
        self.decoder1 = UNetDecoder(32, 16)

        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        """
        Passe l'entrée à travers les encodeurs, le goulot d'étranglement, les décodeurs et applique l'attention.

        Paramètres :
            x (Tensor) : Image d'entrée avec artefacts.

        Retourne :
            Tensor : Image corrigée sans artefacts.
        """
        enc1, x = self.encoder1(x)
        enc2, x = self.encoder2(x)
        enc3, x = self.encoder3(x)
        enc4, x = self.encoder4(x)
        enc5, x = self.encoder5(x)

        x = self.bottleneck(x)

        x = self.decoder5(x, enc5)
        x = self.attention5(x) * x

        x = self.decoder4(x, enc4)
        x = self.attention4(x) * x

        x = self.decoder3(x, enc3)
        x = self.attention3(x) * x

        x = self.decoder2(x, enc2)
        x = self.attention2(x) * x

        x = self.decoder1(x, enc1)
        x = self.attention1(x) * x

        return self.final_conv(x)

    def configure_optimizers(self):
        """
        Configure l'optimiseur et le scheduler pour l'entraînement.

        Retourne :
            tuple : Optimiseur et scheduler configurés.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5),
            'monitor': 'validation_loss',
            'interval': 'epoch',
            'frequency': 10
        }
        return [optimizer], [scheduler]

    def loss(self, outputs, target):
        """
        Calcule la perte combinée entre les sorties et les cibles.

        Paramètres :
            outputs (Tensor) : Sorties du modèle.
            target (Tensor) : Résidus de mouvement cibles.

        Retourne :
            Tensor : Perte combinée MSE et L1.
        """
        mse_loss = F.mse_loss(outputs, target)
        l1_loss = F.l1_loss(outputs, target)
        return 0.7 * l1_loss + 0.3 * mse_loss

    def training_step(self, batch, batch_idx):
        """
        Effectue une étape d'entraînement et calcule la perte.

        Paramètres :
            batch (dict) : Contient les images avec artefacts et les résidus de mouvement.
            batch_idx (int) : Indice du batch.

        Retourne :
            Tensor : Perte calculée pour le batch actuel.
        """
        inputs_avec_artefact = batch['avec_artefact']
        residu_mouvement = batch['residu_mouvement']
        outputs = self(inputs_avec_artefact)

        loss = self.loss(outputs, residu_mouvement)
        self.log('training_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Afficher les prédictions toutes les 100 itérations
        if batch_idx % 100 == 0:
            self.visualize_predictions(inputs_avec_artefact, outputs, residu_mouvement, batch_idx, 'train')

        return loss

    def on_train_epoch_end(self):
        """
        Enregistre la perte d'entraînement à la fin de chaque époque.
        """
        epoch_loss = self.trainer.callback_metrics['training_loss'].item()
        self.train_losses.append(epoch_loss)

    def validation_step(self, batch, batch_idx):
        """
        Effectue une étape de validation et calcule la perte.

        Paramètres :
            batch (dict) : Contient les images avec artefacts et les résidus de mouvement.
            batch_idx (int) : Indice du batch.

        Retourne :
            Tensor : Perte calculée pour le batch actuel.
        """
        inputs_avec_artefact = batch['avec_artefact']
        residu_mouvement = batch['residu_mouvement']
        outputs = self(inputs_avec_artefact)

        loss = self.loss(outputs, residu_mouvement)
        self.log('validation_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Afficher les prédictions au début de la validation
        if batch_idx == 0:
            self.visualize_predictions(inputs_avec_artefact, outputs, residu_mouvement, batch_idx, 'val')

        return loss

    def on_validation_epoch_end(self):
        """
        Enregistre la perte de validation à la fin de chaque époque.
        """
        epoch_loss = self.trainer.callback_metrics['validation_loss'].item()
        self.val_losses.append(epoch_loss)

    def visualize_predictions(self, inputs, outputs, targets, batch_idx, phase):
        """
        Affiche les images d'entrée, les prédictions du modèle (résidus prédits), et les cibles pendant l'entraînement/validation.

        Paramètres :
            inputs (Tensor) : Images d'entrée avec artefacts.
            outputs (Tensor) : Prédictions du modèle (résidus prédits).
            targets (Tensor) : Résidus de mouvement réels.
            batch_idx (int) : Indice du batch.
            phase (str) : Phase actuelle ('train' ou 'val').
        """
        inputs = inputs.cpu().detach().numpy().squeeze()
        outputs = outputs.cpu().detach().numpy().squeeze()
        targets = targets.cpu().detach().numpy().squeeze()

        corrected_image = inputs - outputs

        plt.figure(figsize=(20, 5))

        plt.subplot(1, 4, 1)
        plt.imshow(inputs[0], cmap='gray')
        plt.title('Image avec Artefact')

        plt.subplot(1, 4, 2)
        plt.imshow(outputs[0], cmap='gray')
        plt.title('Prédiction du Modèle')

        plt.subplot(1, 4, 3)
        plt.imshow(targets[0], cmap='gray')
        plt.title('Résidu de Mouvement Réel')

        plt.subplot(1, 4, 4)
        plt.imshow(corrected_image[0], cmap='gray')
        plt.title('Image Corrigée')

        plt.suptitle(f'Phase: {phase} - Batch: {batch_idx}')
        plt.show()

    def plot_loss(self):
        """
        Affiche les courbes de perte d'entraînement et de validation au cours des époques.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curves')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    model = UNet(learning_rate=1e-4)

    early_stopping_callback = EarlyStopping(
        monitor='validation_loss',
        patience=15,
        verbose=True,
    )

    trainer = pl.Trainer(
        callbacks=[early_stopping_callback],
        max_epochs=100,
        precision=16,
        accelerator="gpu",
        devices=1
    )

    trainer.fit(model, train_loader, val_loader)
    print("Entraînement terminé.")

    torch.save(model.state_dict(), "unet_model_residus_mouvements_9.pth")
    print("Modèle U-Net enregistré sous 'unet_model_residus_mouvements_9.pth'")

    model.plot_loss()
