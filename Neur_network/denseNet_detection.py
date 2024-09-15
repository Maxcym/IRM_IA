import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 1. Préparation du Dataset

# Chemin vers le dataset
data_dir = "C:/Users/maxim/PycharmProjects/IRM_IA/Dataset_Normalized_DenseNet"

# Définir les transformations pour les images d'entraînement et de validation
train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convertir en niveaux de gris
    transforms.RandomHorizontalFlip(),  # Appliquer un flip horizontal aléatoire
    transforms.RandomVerticalFlip(),  # Appliquer un flip vertical aléatoire
    transforms.RandomRotation(20),  # Appliquer une rotation aléatoire entre -20 et 20 degrés
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Recadrage aléatoire et redimensionnement
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Jitter des couleurs
    transforms.ToTensor(),  # Convertir les images en tenseurs PyTorch
])

val_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convertir en niveaux de gris
    transforms.ToTensor(),  # Convertir les images en tenseurs PyTorch
])

# Charger le dataset à partir des répertoires
full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)

# Taille du dataset
total_size = len(full_dataset)  # Nombre total d'images dans le dataset
train_size = int(0.8 * total_size)  # 80% pour l'entraînement
val_size = total_size - train_size  # 20% pour la validation

# Diviser le dataset en ensembles d'entraînement et de validation
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Appliquer les transformations spécifiques à l'ensemble de validation
val_dataset.dataset.transform = val_transforms

# Créer les DataLoaders pour l'entraînement et la validation
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)  # DataLoader pour l'entraînement
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)  # DataLoader pour la validation


# 2. Définition du Modèle DenseNet sans Bottleneck

class DenseLayer(nn.Module):
    """
    Couche dense du modèle DenseNet sans bottleneck.

    Attributs :
        bn (nn.BatchNorm2d) : Normalisation de batch pour stabiliser l'entraînement.
        conv (nn.Conv2d) : Convolution 3x3 pour extraire les caractéristiques.

    Méthodes :
        forward(x) : Applique BatchNorm, ReLU, Conv3x3 et concatène la sortie avec l'entrée.
    """

    def __init__(self, in_channels, growth_rate):
        """
        Initialise une couche dense avec les canaux d'entrée et le taux de croissance spécifiés.

        Paramètres :
            in_channels (int) : Nombre de canaux en entrée.
            growth_rate (int) : Nombre de canaux ajoutés par la couche dense.
        """
        super(DenseLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)  # BatchNorm pour stabiliser l'entraînement
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)  # Convolution 3x3

    def forward(self, x):
        """
        Applique la normalisation, l'activation et la convolution, puis concatène la sortie avec l'entrée.

        Paramètres :
            x (Tensor) : Entrée du tenseur.

        Retourne :
            Tensor : Résultat de la couche dense avec concaténation.
        """
        out = self.conv(F.relu(self.bn(x)))  # Passer par BatchNorm, ReLU et Conv3x3
        out = torch.cat([out, x], 1)  # Concaténer la sortie avec l'entrée pour les connexions denses
        return out


class DenseBlock(nn.Module):
    """
    Bloc dense composé de plusieurs couches denses.

    Attributs :
        denseblock (nn.Sequential) : Séquence de couches denses.

    Méthodes :
        forward(x) : Passe l'entrée à travers toutes les couches denses du bloc.
    """

    def __init__(self, num_layers, in_channels, growth_rate):
        """
        Initialise un bloc dense avec un nombre spécifié de couches denses.

        Paramètres :
            num_layers (int) : Nombre de couches denses dans le bloc.
            in_channels (int) : Nombre de canaux en entrée du bloc.
            growth_rate (int) : Nombre de canaux ajoutés par chaque couche dense.
        """
        super(DenseBlock, self).__init__()
        layers = [DenseLayer(in_channels + i * growth_rate, growth_rate) for i in range(num_layers)]
        self.denseblock = nn.Sequential(*layers)

    def forward(self, x):
        """
        Applique toutes les couches denses du bloc à l'entrée.

        Paramètres :
            x (Tensor) : Entrée du tenseur.

        Retourne :
            Tensor : Sortie du bloc dense.
        """
        return self.denseblock(x)


class TransitionLayer(nn.Module):
    """
    Couche de transition pour réduire le nombre de canaux entre deux blocs denses.

    Attributs :
        bn (nn.BatchNorm2d) : Normalisation de batch.
        conv (nn.Conv2d) : Convolution 1x1 pour réduire le nombre de canaux.
        pool (nn.AvgPool2d) : Pooling pour réduire la taille spatiale.

    Méthodes :
        forward(x) : Applique BatchNorm, ReLU, Conv1x1 et pooling.
    """

    def __init__(self, in_channels, out_channels):
        """
        Initialise une couche de transition pour réduire le nombre de canaux.

        Paramètres :
            in_channels (int) : Nombre de canaux en entrée.
            out_channels (int) : Nombre de canaux en sortie après la réduction.
        """
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """
        Applique la réduction des canaux et la réduction de la taille spatiale.

        Paramètres :
            x (Tensor) : Entrée du tenseur.

        Retourne :
            Tensor : Sortie après réduction des canaux et pooling.
        """
        out = self.conv(F.relu(self.bn(x)))
        out = self.pool(out)
        return out


class CustomDenseNet(pl.LightningModule):
    """
    Modèle DenseNet personnalisé sans bottleneck, implémenté avec PyTorch Lightning.

    Attributs :
        learning_rate (float) : Taux d'apprentissage pour l'optimisation.
        train_losses (list) : Liste des pertes d'entraînement par époque.
        val_losses (list) : Liste des pertes de validation par époque.

    Méthodes :
        forward(x) : Passe l'entrée à travers toutes les couches du modèle.
        configure_optimizers() : Configure l'optimiseur et le scheduler.
        training_step(batch, batch_idx) : Effectue une étape d'entraînement et calcule la perte.
        on_train_epoch_end() : Enregistre la perte d'entraînement à la fin de chaque époque.
        validation_step(batch, batch_idx) : Effectue une étape de validation et calcule la perte.
        on_validation_epoch_end() : Enregistre la perte de validation à la fin de chaque époque.
        plot_loss(dataset_name, threshold) : Affiche la courbe des pertes d'entraînement et de validation.
    """

    def __init__(self, growth_rate=8, block_layers=[4, 8, 12, 8], num_classes=2, learning_rate=0.001):
        """
        Initialise le modèle DenseNet avec les paramètres spécifiés.

        Paramètres :
            growth_rate (int) : Taux de croissance des canaux dans les blocs denses.
            block_layers (list) : Liste du nombre de couches dans chaque bloc dense.
            num_classes (int) : Nombre de classes de sortie pour la classification.
            learning_rate (float) : Taux d'apprentissage pour l'entraînement.
        """
        super(CustomDenseNet, self).__init__()
        self.learning_rate = learning_rate
        self.train_losses = []
        self.val_losses = []
        num_channels = 2 * growth_rate

        self.conv1 = nn.Conv2d(1, num_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.blocks = nn.ModuleList()
        for i, num_layers in enumerate(block_layers):
            block = DenseBlock(num_layers, num_channels, growth_rate)
            self.blocks.append(block)
            num_channels += num_layers * growth_rate
            if i != len(block_layers) - 1:
                trans = TransitionLayer(num_channels, num_channels // 2)
                self.blocks.append(trans)
                num_channels = num_channels // 2

        self.bn2 = nn.BatchNorm2d(num_channels)
        self.dropout = nn.Dropout(p=0)
        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x):
        """
        Passe l'entrée à travers toutes les couches du modèle pour générer une prédiction.

        Paramètres :
            x (Tensor) : Entrée du tenseur.

        Retourne :
            Tensor : Sortie du modèle.
        """
        out = self.pool1(F.relu(self.bn1(self.conv1(x))))
        for block in self.blocks:
            out = block(out)
        out = F.relu(self.bn2(out))
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

    def configure_optimizers(self):
        """
        Configure l'optimiseur et le scheduler pour l'entraînement.

        Retourne :
            tuple : Optimiseur et scheduler pour l'entraînement.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0)
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5),
            'monitor': 'validation_loss',
            'interval': 'epoch',
            'frequency': 10
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        """
        Effectue une étape d'entraînement et calcule la perte.

        Paramètres :
            batch (tuple) : Données d'entraînement (entrée, étiquettes).
            batch_idx (int) : Indice du batch.

        Retourne :
            Tensor : Perte calculée pour le batch actuel.
        """
        inputs, labels = batch
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        self.log('training_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log('training_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
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
            batch (tuple) : Données de validation (entrée, étiquettes).
            batch_idx (int) : Indice du batch.

        Retourne :
            Tensor : Perte calculée pour le batch actuel.
        """
        inputs, labels = batch
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        self.log('validation_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log('validation_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_end(self):
        """
        Enregistre la perte de validation à la fin de chaque époque.
        """
        epoch_loss = self.trainer.callback_metrics['validation_loss'].item()
        self.val_losses.append(epoch_loss)

    def plot_loss(self, dataset_name="Dataset", threshold=0.5):
        """
        Affiche la courbe des pertes d'entraînement et de validation.

        Paramètres :
            dataset_name (str) : Nom du dataset utilisé pour l'entraînement.
            threshold (float) : Seuil pour la courbe des pertes.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Loss Curve - Dataset: {dataset_name}, Threshold: {threshold}')
        plt.legend()
        plt.show()


# 3. Entraînement du Modèle
if __name__ == '__main__':
    model = CustomDenseNet(num_classes=2, learning_rate=3e-6)

    early_stopping_callback = EarlyStopping(
        monitor='validation_loss',
        patience=3,
        verbose=True,
    )

    trainer = Trainer(callbacks=[early_stopping_callback], max_epochs=100)

    trainer.fit(model, train_loader, val_loader)

    model.plot_loss(dataset_name="CustomDataset", threshold=0.5)

    torch.save(model.state_dict(), "simplified_densenet_model.pth")
    print("Modèle simplifié enregistré sous 'simplified_densenet_model.pth'")
