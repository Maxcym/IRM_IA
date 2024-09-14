# Détection et Correction des Artefacts de Mouvement sur les Images d'IRM Cérébrales

## Description du Projet

Ce projet vise à développer des méthodes basées sur l'intelligence artificielle pour détecter et corriger les artefacts de mouvement sur les images d'Imagerie par Résonance Magnétique (IRM) cérébrales. Les artefacts de mouvement, causés par des déplacements volontaires ou involontaires des patients lors des acquisitions IRM, altèrent la qualité des images et compliquent le diagnostic médical. En s'appuyant sur des réseaux de neurones profonds tels que DenseNet pour la détection et U-Net pour la correction, ce projet propose une solution pour automatiser le processus de correction des images, améliorant ainsi leur qualité et facilitant l'interprétation clinique.

## Objectifs du Projet

- Étudier les artefacts de mouvement et leur impact sur les images IRM cérébrales.
- Développer un algorithme de détection des artefacts de mouvement à l'aide de réseaux de neurones convolutifs.
- Concevoir un modèle pour corriger ces artefacts.

## Méthodologie

1. **Étude des Artefacts de Mouvement** : Analyse des types d'artefacts causés par les mouvements et de leur impact sur les images IRM.
2. **Développement de Modèles** : Utilisation de DenseNet pour détecter les artefacts et de U-Net pour corriger les images dégradées. Les modèles sont entraînés sur des jeux de données comprenant des images IRM avec et sans artefacts.
3. **Entraînement et Évaluation** : Les réseaux de neurones sont entraînés sur des serveurs équipés de GPU pour maximiser l'efficacité du traitement. Les performances sont évaluées à l'aide de métriques comme le SSIM.
4. **Optimisation** : Ajustement des hyperparamètres via des techniques d'optimisation bayésienne pour améliorer la précision et la fiabilité des modèles.

## Technologies Utilisées

- **Python** : Langage principal pour le développement des algorithmes.
- **PyTorch** : Framework utilisé pour l'implémentation des réseaux de neurones.
- **Optuna** : Bibliothèque pour l'optimisation des hyperparamètres.

## Datasets d'Entraînement

Le dataset est composé de 9600 IRM cérébrales avec et sans artefacts de mouvement. Les images sans artefacts proviennent des datasets publics **OpenNEURO** et **IXI Dataset**. Ces images incluent des acquisitions T1, T2 et PD et couvrent différents angles de vue : sagittal, axial et coronal, ce qui permet de diversifier le dataset pour une meilleure généralisation des modèles. 

- **Images sans artefacts** : Utilisées comme référence pour évaluer les performances des modèles.
- **Images avec artefacts de mouvement** : Générées en simulant des décalages de phase linéaires aléatoires dans l’espace K, reproduisant les effets de mouvements réels des patients.

Pour visualiser la diversité des artefacts de mouvement dans ce dataset, une analyse de leur distribution a été réalisée en utilisant la métrique **SSIM (Structural Similarity Index)**, qui mesure l'intensité des artefacts par rapport à une image non corrompue. Les valeurs obtenues ont été exploitées pour définir des seuils à l'aide des quartiles :
- Le premier quartile représente les images les plus dégradées.
- La médiane indique un niveau modéré d’artefacts.
- Le troisième quartile correspond aux IRM ayant le moins d’artefacts.

Cette approche permet de mieux comprendre la diversité et la sévérité des artefacts présents dans les images et d’ajuster les modèles pour qu’ils soient plus robustes face à différents niveaux de dégradation.


## Hyperparamètres des Modèles

### Hyperparamètres du modèle DenseNet 

- **Taux de croissance (« growth_rate » = 8)** : Contrôle le nombre de canaux après chaque couche dans les blocs denses.
- **Nombre de couches dans chaque bloc (« block_layers » = [4,8,12,8])** : Définit le nombre de couches de convolution 3x3 dans chaque bloc dense.
- **Nombre de classes (« num_classes » = 2)** : Correspond au nombre de sorties de la couche fully connected, soit le nombre de classes pour la classification.
- **Taux de Dropout (« dropout » = 0)** : Définit la probabilité pour chaque neurone d'être désactivé pendant l'entraînement pour réduire le risque de surapprentissage.
- **Taux d’apprentissage (« learning_rate » = 3x10⁻⁶)** : Indique la vitesse à laquelle les poids sont mis à jour pendant l’entraînement.
- **Pondération L2 (« weight_decay » = 0)** : Pénalise les grands poids en ajoutant la somme des carrés des poids à la fonction perte, réduisant le risque de surapprentissage.
- **Nombre d’époques / epoch (« max_epochs » = 100)** : Nombre maximal de passes sur l’ensemble des données d’entraînement.
- **Batch size (« batch_size » = 128)** : Nombre d’exemples d’entraînement utilisés pour calculer le gradient lors d’une passe d’entraînement.
- **Taille du Kernel (« kernel_size »)** : Dimension du noyau de convolution utilisé pendant l’opération de convolution.
- **Arrêt prématuré / Early stopping (« patience » = 3)** : Arrête l’entraînement lorsque le loss de validation reste constant ou augmente sur un nombre d'epochs définis.
- **Stride (« Stride »)** : Pas de déplacement du filtre de convolution à travers l’image d’entrée.
- **Padding (« padding »)** : Ajoute des pixels supplémentaires autour des bords de l’image avant l’application du filtre de convolution pour maintenir la taille des dimensions spatiales de la sortie.
- **Scheduler de réduction du taux d'apprentissage (« ReduceLROnPlateau »)** : Réduit le taux d'apprentissage lorsque la métrique surveillée reste constante.

### Hyperparamètres du modèle U-Net 

- **Learning Rate (« learning_rate » = 1x10⁻⁴)** : Indique la vitesse à laquelle les poids sont mis à jour pendant l’entraînement.
- **Nombre de canaux des couches convolutives (« in_channels », « out_channels »)** : Définit le nombre de canaux en entrée et en sortie pour chaque couche convolutionnelle.
- **Kernel Size (« kernel_size »)** : Dimension du noyau de convolution utilisé pendant l’opération de convolution.
- **Padding (« padding »)** : Ajoute des pixels supplémentaires autour des bords de l’image avant l’application du filtre de convolution.
- **Stride (« stride »)** : Pas de déplacement du filtre de convolution à travers l’image d’entrée.
- **Nombre d’époques (« max_epochs » = 100)** : Nombre maximum de passes sur l'ensemble des données d'entraînement.
- **Batch Size (« batch_size » = 256)** : Nombre d’exemples d’entraînement utilisés pour calculer le gradient lors d’une passe d’entraînement.
- **Scheduler de réduction du taux d'apprentissage (« ReduceLROnPlateau »)** : Réduit le taux d'apprentissage lorsqu’une métrique surveillée reste constante.
- **Early Stopping (« patience » = 15)** : Interrompt l'entraînement lorsque la perte de validation ne s’améliore pas sur un nombre d'époques défini.
- **Attention Kernel Size (« kernel_size » = 7)** : Taille du noyau utilisé dans le module d'attention spatiale pour capturer les caractéristiques des cartes de caractéristiques.

