import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam, AdamW
import optuna
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
import optuna.visualization as vis
from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt

torch.set_float32_matmul_precision('high')

# 1. Préparation du Dataset
data_dir = "/homes_unix/cayman/PycharmProjects/artefact_detection/Dataset_Normalized"

# Définir les transformations pour les images d'entraînement et de validation
train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
])

val_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

# Charger le dataset
full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)
total_size = len(full_dataset)
train_size = int(0.8 * total_size)
val_size = total_size - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
val_dataset.dataset.transform = val_transforms

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

# 2. Définition du Modèle DenseNet sans Bottleneck
class DenseLayer(nn.Module):
    """
    Définit une couche dense avec Batch Normalization et une convolution 3x3.

    Attributs :
        bn (nn.BatchNorm2d) : Normalisation par lot pour stabiliser l'entraînement.
        conv (nn.Conv2d) : Convolution 3x3 pour extraire les caractéristiques.

    Méthodes :
        forward(x) : Applique la normalisation, l'activation ReLU, et la convolution.
    """
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = torch.cat([out, x], 1)
        return out

class DenseBlock(nn.Module):
    """
    Bloc dense composé de plusieurs couches denses.

    Attributs :
        denseblock (nn.Sequential) : Séquence de couches denses.

    Méthodes :
        forward(x) : Passe l'entrée à travers les couches denses.
    """
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        layers = [DenseLayer(in_channels + i * growth_rate, growth_rate) for i in range(num_layers)]
        self.denseblock = nn.Sequential(*layers)

    def forward(self, x):
        return self.denseblock(x)

class TransitionLayer(nn.Module):
    """
    Couche de transition pour réduire le nombre de canaux après un bloc dense.

    Attributs :
        bn (nn.BatchNorm2d) : Normalisation par lot.
        conv (nn.Conv2d) : Convolution 1x1 pour réduire les dimensions des canaux.
        pool (nn.AvgPool2d) : Pooling moyen pour réduire la taille des caractéristiques.

    Méthodes :
        forward(x) : Applique la normalisation, l'activation ReLU, la convolution et le pooling.
    """
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = self.pool(out)
        return out

class CustomDenseNet(pl.LightningModule):
    """
    Modèle DenseNet personnalisé pour la classification.

    Attributs :
        learning_rate (float) : Taux d'apprentissage.
        optimizer_name (str) : Nom de l'optimiseur (Adam ou AdamW).
        factor (float) : Facteur de réduction du taux d'apprentissage.
        patience (int) : Nombre d'époques d'attente avant de réduire le taux d'apprentissage.
        train_losses (list) : Liste des pertes d'entraînement.
        val_losses (list) : Liste des pertes de validation.
        conv1, bn1, pool1 (nn.Modules) : Couches d'entrée du réseau.
        blocks (nn.ModuleList) : Liste de blocs denses et de couches de transition.
        bn2, fc (nn.Modules) : Couches de sortie du réseau.

    Méthodes :
        forward(x) : Applique les transformations du modèle sur les données d'entrée.
        configure_optimizers() : Configure l'optimiseur et le scheduler.
        training_step(batch, batch_idx) : Effectue une étape d'entraînement.
        validation_step(batch, batch_idx) : Effectue une étape de validation.
    """
    def __init__(self, growth_rate=8, block_layers=[4, 8, 12, 8], num_classes=2, learning_rate=0.001, optimizer_name='Adam', factor=0.1, patience=5):
        super(CustomDenseNet, self).__init__()
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.factor = factor
        self.patience = patience
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
        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x):
        out = self.pool1(F.relu(self.bn1(self.conv1(x))))
        for block in self.blocks:
            out = block(out)
        out = F.relu(self.bn2(out))
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return self.fc(out)

    def configure_optimizers(self):
        if self.optimizer_name == 'Adam':
            optimizer = Adam(self.parameters(), lr=self.learning_rate)
        else:
            optimizer = AdamW(self.parameters(), lr=self.learning_rate)

        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=self.factor, patience=self.patience),
            'monitor': 'validation_loss'
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        self.log('training_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        self.log('validation_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

class CustomPruningCallback(Callback):
    """
    Callback personnalisé pour Optuna, permettant de stopper l'entraînement si les performances stagnent.

    Attributs :
        trial (optuna.Trial) : L'essai en cours d'Optuna.
        monitor (str) : La métrique à surveiller pour décider du pruning.

    Méthodes :
        on_validation_end(trainer, pl_module) : Vérifie la métrique de validation et applique le pruning si nécessaire.
    """
    def __init__(self, trial, monitor="validation_loss"):
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def on_validation_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        current_score = logs.get(self.monitor)

        if current_score is None:
            return

        self.trial.report(current_score, step=trainer.current_epoch)

        if self.trial.should_prune():
            raise optuna.exceptions.TrialPruned()

# 3. Fonction Objective pour Optuna
def objective(trial):
    """
    Fonction objective pour l'optimisation avec Optuna.

    Paramètres :
        trial (optuna.Trial) : Un essai pour tester un ensemble d'hyperparamètres.

    Retourne :
        float : La perte de validation obtenue avec les hyperparamètres testés.
    """
    learning_rate = trial.suggest_float('learning_rate', 1e-7, 1e-3, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW'])
    factor = trial.suggest_categorical('factor', [0.1, 0.2])
    patience = trial.suggest_int('patience', 3, 10)

    model = CustomDenseNet(
        learning_rate=learning_rate,
        optimizer_name=optimizer_name,
        factor=factor,
        patience=patience
    )

    pruning_callback = CustomPruningCallback(trial, monitor="validation_loss")

    trainer = Trainer(
        max_epochs=100,
        callbacks=[pruning_callback],
        precision="16-mixed",
        accelerator="gpu",
        devices=1,
    )

    trainer.fit(model, train_loader, val_loader)

    val_loss = trainer.callback_metrics["validation_loss"].item()
    return val_loss

# 4. Étude Optuna
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

# Affichage des meilleurs paramètres et des visualisations
print("Meilleurs paramètres : ", study.best_params)

fig1 = vis.plot_optimization_history(study)
fig1.show()

fig2 = vis.plot_param_importances(study)
fig2.show()

fig3 = vis.plot_slice(study)
fig3.show()

fig4 = vis.plot_parallel_coordinate(study)
fig4.show()
