import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import optuna
from pytorch_lightning.loggers import CSVLogger

torch.set_float32_matmul_precision('high')


class ThreeClassDataset(Dataset):
    """
    Dataset pour les images avec et sans artefacts de mouvement.

    Attributs :
        root_dir (str) : Dossier racine contenant les sous-dossiers des classes.
        transform (callable, optionnel) : Transformations appliquées sur les images.
        sans_artefacts_dir (str) : Dossier contenant les images sans artefacts.
        avec_artefacts_dir (str) : Dossier contenant les images avec artefacts.
        sans_artefacts_files (list) : Liste des fichiers sans artefacts.
        avec_artefacts_files (list) : Liste des fichiers avec artefacts.

    Méthodes :
        __len__() : Retourne le nombre d'images dans le dataset.
        __getitem__(idx) : Retourne une paire d'images (avec et sans artefacts) à l'index donné.
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.sans_artefacts_dir = os.path.join(root_dir, "Class_0")
        self.avec_artefacts_dir = os.path.join(root_dir, "Class_1")
        self.sans_artefacts_files = sorted(os.listdir(self.sans_artefacts_dir))
        self.avec_artefacts_files = sorted(os.listdir(self.avec_artefacts_dir))
        assert len(self.sans_artefacts_files) == len(self.avec_artefacts_files), \
            "Les nombres d'images doivent correspondre pour chaque classe."

    def __len__(self):
        return len(self.sans_artefacts_files)

    def __getitem__(self, idx):
        sans_artefact_path = os.path.join(self.sans_artefacts_dir, self.sans_artefacts_files[idx])
        avec_artefact_path = os.path.join(self.avec_artefacts_dir, self.avec_artefacts_files[idx])
        sans_artefact_image = Image.open(sans_artefact_path)
        avec_artefact_image = Image.open(avec_artefact_path)
        if self.transform:
            sans_artefact_image = self.transform(sans_artefact_image)
            avec_artefact_image = self.transform(avec_artefact_image)
        return {
            'sans_artefact': sans_artefact_image,
            'avec_artefact': avec_artefact_image
        }


# Définition des transformations pour l'entraînement et la validation
train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
])

val_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

# Préparation du dataset
data_dir = "C:/Users/maxim/PycharmProjects/artefact_detection/Dataset_Normalized"
full_dataset = ThreeClassDataset(root_dir=data_dir, transform=train_transforms)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
val_dataset.dataset.transform = val_transforms

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)


class ConvBlock(nn.Module):
    """
    Bloc de convolution avec Batch Normalization.

    Attributs :
        conv (nn.Sequential) : Séquence de convolutions, normalisation et activation ReLU.

    Méthodes :
        forward(x) : Passe avant du bloc de convolution.
    """

    def __init__(self, in_channels, out_channels):
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
        return self.conv(x)


class UNetEncoder(nn.Module):
    """
    Encodeur pour le modèle U-Net, comprenant une convolution et une mise en commun.

    Attributs :
        conv (ConvBlock) : Bloc de convolution.
        pool (nn.MaxPool2d) : Opération de mise en commun pour réduire la taille des caractéristiques.

    Méthodes :
        forward(x) : Passe avant de l'encodeur.
    """

    def __init__(self, in_channels, out_channels):
        super(UNetEncoder, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x_pooled = self.pool(x)
        return x, x_pooled


class UNetDecoder(nn.Module):
    """
    Décodeur pour le modèle U-Net, comprenant une convolution transposée et une concaténation.

    Attributs :
        upconv (nn.ConvTranspose2d) : Convolution transposée pour augmenter la taille des caractéristiques.
        conv (ConvBlock) : Bloc de convolution pour affiner les caractéristiques après concaténation.

    Méthodes :
        forward(x, encoder_features) : Passe avant du décodeur.
    """

    def __init__(self, in_channels, out_channels):
        super(UNetDecoder, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + out_channels, out_channels)

    def forward(self, x, encoder_features):
        x = self.upconv(x)
        diffY = encoder_features.size()[2] - x.size()[2]
        diffX = encoder_features.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat((x, encoder_features), dim=1)
        return self.conv(x)


class UNet(pl.LightningModule):
    """
    Modèle U-Net pour la correction des artefacts de mouvement dans les images IRM.

    Attributs :
        learning_rate (float) : Taux d'apprentissage.
        weight_decay (float) : Décroissance du poids pour régulariser le modèle.
        train_losses (list) : Liste des pertes d'entraînement.
        val_losses (list) : Liste des pertes de validation.
        encoder1-4 (UNetEncoder) : Encodeurs du modèle U-Net.
        decoder1-4 (UNetDecoder) : Décodeurs du modèle U-Net.
        bottleneck (ConvBlock) : Bloc de goulot d'étranglement pour la partie la plus basse du réseau.
        final_conv (nn.Conv2d) : Convolution finale pour produire l'image corrigée.
        mse_loss (nn.MSELoss) : Fonction de perte pour calculer l'erreur quadratique moyenne.

    Méthodes :
        forward(x) : Passe avant du modèle.
        configure_optimizers() : Configure l'optimiseur pour l'entraînement.
        training_step(batch, batch_idx) : Effectue une étape d'entraînement et enregistre la perte.
        validation_step(batch, batch_idx) : Effectue une étape de validation et enregistre la perte.
        on_train_epoch_end() : Enregistre la perte d'entraînement à la fin de chaque époque.
        on_validation_epoch_end() : Enregistre la perte de validation à la fin de chaque époque.
        plot_loss(dataset_name) : Affiche les courbes de perte d'entraînement et de validation.
    """

    def __init__(self, learning_rate=0.001, weight_decay=0, filters=[8, 16, 32, 64, 128]):
        super(UNet, self).__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.train_losses = []
        self.val_losses = []

        self.encoder1 = UNetEncoder(1, filters[0])
        self.encoder2 = UNetEncoder(filters[0], filters[1])
        self.encoder3 = UNetEncoder(filters[1], filters[2])
        self.encoder4 = UNetEncoder(filters[2], filters[3])

        self.bottleneck = ConvBlock(filters[3], filters[4])

        self.decoder4 = UNetDecoder(filters[4], filters[3])
        self.decoder3 = UNetDecoder(filters[3], filters[2])
        self.decoder2 = UNetDecoder(filters[2], filters[1])
        self.decoder1 = UNetDecoder(filters[1], filters[0])

        self.final_conv = nn.Conv2d(filters[0], 1, kernel_size=1)
        self.mse_loss = nn.MSELoss()

    def forward(self, x):
        enc1, x = self.encoder1(x)
        enc2, x = self.encoder2(x)
        enc3, x = self.encoder3(x)
        enc4, x = self.encoder4(x)
        x = self.bottleneck(x)
        x = self.decoder4(x, enc4)
        x = self.decoder3(x, enc3)
        x = self.decoder2(x, enc2)
        x = self.decoder1(x, enc1)
        return self.final_conv(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'validation_loss'
            }
        }

    def training_step(self, batch, batch_idx):
        inputs_avec_artefact = batch['avec_artefact']
        target = batch['sans_artefact']
        outputs = self(inputs_avec_artefact)
        loss = self.mse_loss(outputs, target)
        self.log('training_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs_avec_artefact = batch['avec_artefact']
        target = batch['sans_artefact']
        outputs = self(inputs_avec_artefact)
        loss = self.mse_loss(outputs, target)
        self.log('validation_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        epoch_loss = self.trainer.callback_metrics.get('training_loss', None)
        if epoch_loss is not None:
            self.train_losses.append(epoch_loss.item())
        else:
            print("Training loss not recorded for this epoch.")

    def on_validation_epoch_end(self):
        epoch_loss = self.trainer.callback_metrics.get('validation_loss', None)
        if epoch_loss is not None:
            self.val_losses.append(epoch_loss.item())
        else:
            print("Validation loss not recorded for this epoch.")

    def plot_loss(self, dataset_name="Dataset"):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Loss Curve - {dataset_name}')
        plt.legend()
        plt.show()


# Fonction d'optimisation avec Optuna
def objective(trial):
    """
    Fonction d'objectif pour l'optimisation avec Optuna.

    Paramètres :
        trial (optuna.Trial) : Un essai pour tester un ensemble d'hyperparamètres.

    Retourne :
        float : La perte de validation obtenue avec les hyperparamètres testés.
    """
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW'])
    factor = trial.suggest_float('factor', 0.1, 0.5)
    patience = trial.suggest_int('patience', 3, 10)

    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam
    else:
        optimizer = torch.optim.AdamW

    model = UNet(learning_rate=learning_rate)

    # Configuration de l'optimiseur avec le scheduler de réduction de taux d'apprentissage
    model.configure_optimizers = lambda: {
        'optimizer': optimizer(model.parameters(), lr=learning_rate),
        'lr_scheduler': {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer(model.parameters(), lr=learning_rate),
                factor=factor,
                patience=patience
            ),
            'monitor': 'validation_loss'
        }
    }

    logger = CSVLogger("logs", name=f"trial_{trial.number}")

    trainer = Trainer(
        max_epochs=10,
        logger=logger,
        enable_checkpointing=False
    )

    trainer.fit(model, train_loader, val_loader)

    return trainer.callback_metrics['validation_loss'].item()


if __name__ == '__main__':
    # Création d'une étude Optuna pour minimiser la perte de validation
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)

    print('Best trial:')
    trial = study.best_trial
    print(f'  Validation Loss: {trial.value}')
    print(f'  Best hyperparameters: {trial.params}')

    best_model = UNet(
        learning_rate=trial.params['learning_rate']
    )

    early_stopping_callback = EarlyStopping(
        monitor='validation_loss',
        patience=trial.params['patience'],
        verbose=True,
    )

    trainer = Trainer(
        max_epochs=50,
        callbacks=[early_stopping_callback],
        precision=16,
        accelerator="gpu",
        devices=1
    )
    trainer.fit(best_model, train_loader, val_loader)

    best_model.plot_loss(dataset_name="MRI Artefact Correction with MSE Loss")
    torch.save(best_model.state_dict(), "best_unet_model.pth")
    print("Modèle U-Net final enregistré sous 'best_unet_model.pth'")
