# Détection et correction des artefacts de mouvement sur les images d'IRM cérébrales

## Description du projet

Ce projet vise à développer des méthodes basées sur l'intelligence artificielle pour détecter et corriger les artefacts de mouvement sur les images d'Imagerie par Résonance Magnétique (IRM) cérébrales. Ces artefacts sont causés par des déplacements volontaires ou involontaires des patients lors des acquisitions IRM, ce qui dégrade la qualité des images et complique le diagnostic médical. Ce travail propose donc une solution visant à améliorer la qualité des images et à faciliter leur interprétation clinique.

## Technologies utilisées

- **Python** : Langage principal pour le développement des algorithmes.
- **PyTorch** : Framework utilisé pour l'implémentation des réseaux de neurones, facilitant la création et l'entraînement de modèles de deep learning.
- **PyTorch Lightning** : Extension de PyTorch qui simplifie le code de recherche en structurant les scripts d'entraînement, rendant le code plus propre et plus facile à gérer tout en permettant une intégration facile avec des outils comme Optuna pour l'optimisation.
- **Optuna** : Bibliothèque pour l'optimisation des hyperparamètres, permettant de trouver les configurations optimales pour améliorer les performances des modèles.
- **PyQt5** : Utilisé pour développer l'interface utilisateur graphique (GUI) permettant de tester les réseaux de neurones pour la détection et correction des artefacts.
- **Scikit-learn** : Employé pour diverses métriques d'évaluation des modèles et la gestion des jeux de données.
- **PIL (Python Imaging Library)** : Utilisée pour le traitement et la manipulation des images dans le pipeline de correction et détection.

## Dataset d'entraînement

Le dataset est composé de 9600 IRM cérébrales avec et sans artefacts de mouvement. Les images sans artefacts proviennent des datasets publics **OpenNEURO** et **IXI Dataset**. Ces images incluent des acquisitions T1, T2 et PD et couvrent différents angles de vue : sagittal, axial et coronal, ce qui permet de diversifier le dataset pour une meilleure généralisation des modèles. 

- **Images sans artefacts** : Utilisées comme référence pour évaluer les performances des modèles.
- **Images avec artefacts de mouvement** : Générées en simulant des décalages de phase linéaires aléatoires dans l’espace K, reproduisant les effets de mouvements réels des patients.

Pour visualiser la diversité des artefacts de mouvement dans ce dataset, une analyse de leur distribution a été réalisée en utilisant la métrique **SSIM (Structural Similarity Index)**, qui mesure l'intensité des artefacts par rapport à une image non corrompue. Les valeurs obtenues ont été exploitées pour définir des seuils à l'aide des quartiles :
- Le premier quartile représente les images les plus dégradées.
- La médiane indique un niveau modéré d’artefacts.
- Le troisième quartile correspond aux IRM ayant le moins d’artefacts.

Cette approche permet de mieux comprendre la diversité et la sévérité des artefacts présents dans les images et d’ajuster les modèles pour qu’ils soient plus robustes face à différents niveaux de dégradation.

## Méthode de simulation des artefacts de mouvement

La simulation des artefacts de mouvement dans ce projet est réalisée en appliquant des décalages de phase aléatoires dans l'espace k des images IRM. Cette méthode permet de reproduire les effets de mouvement observés lors de l'acquisition des images, altérant ainsi la qualité des images pour tester et évaluer les algorithmes de correction.

### Classe `MotionArtifactSimulator`

La classe `MotionArtifactSimulator` permet de simuler des artefacts de mouvement sur des images IRM en introduisant des décalages de phase dans des régions spécifiques de l'espace k. Voici les principales fonctionnalités de cette méthode :

**Simulation des artefacts** :
   - Le simulateur lit les images IRM d'entrée et alterne les effets de décalage entre les directions AP et LR pour chaque  image :
     - **Antéro-Postérieure (AP)** : Décalage dans les lignes de l'espace k.
     - **Latérale (LR)** : Décalage dans les colonnes de l'espace k.
   - Des décalages de phase aléatoires sont appliqués sur des régions spécifiques (centre ou bords) de l'espace k, modifiant l'image en conséquence.

**Enregistrement des décalages de phase sous forme de graphiques** :
   - Les décalages de phase appliqués sont enregistrés dans un fichier CSV pour chaque image, ce qui permet une analyse précise des modifications effectuées.
   - En plus des fichiers CSV, les décalages de phase appliqués sont également enregistrés sous forme de graphiques. Ces graphiques montrent la distribution des décalages dans l'espace k, ce qui permet de visualiser l'impact du mouvement simulé sur l'image.
   - Les images corrompues sont sauvegardées avec leurs résidus de mouvement, qui représentent les différences entre les images d'origine et les images altérées.

**Affichage des Résultats** :
   - Le simulateur permet d'afficher les images originales, les images corrompues, les décalages de phase appliqués, et les résidus de mouvement pour une analyse visuelle des artefacts simulés.

## Optimisation des hyperparamètres

Pour le modèle U-Net, l’optimisation des hyperparamètres a été réalisée pour améliorer l'efficacité et la performance du processus d'entraînement. L'objectif était de trouver les valeurs optimales pour le taux d’apprentissage, le choix de l'optimiseur, le facteur de réduction du taux d’apprentissage, et la patience du scheduler. L'optimisation des hyperparamètres n’a pas été effectuée sur la structure du réseau elle-même, car l'architecture montrait déjà de bons résultats lors des tests préliminaires.

### Stratégie d'optimisation

**Hyperparamètres optimisés** :
   - **Taux d’apprentissage (learning rate)** : Contrôle la vitesse d'ajustement des poids du réseau au cours de l'entraînement. Une valeur trop élevée peut entraîner un apprentissage trop rapide et instable, tandis qu'une valeur trop faible peut ralentir considérablement le processus d'entraînement.
   - **Choix de l’optimiseur (Adam, AdamW)** : Les optimisateurs influencent la façon dont les poids sont mis à jour pendant l'entraînement. Adam et AdamW sont réputés pour leur efficacité et leur capacité à converger rapidement.
   - **Facteur de réduction du taux d’apprentissage** : Réduit le learning rate lorsque la métrique surveillée stagne, ce qui permet d'affiner l'apprentissage du modèle en phase de convergence.
   - **Patience du scheduler** : Définit le nombre d'époques pendant lesquelles le modèle peut stagner avant de réduire le learning rate. Cela aide à éviter la réduction prématurée du taux d'apprentissage.

### Méthodologie d'Optimisation avec Optuna

Optuna a été utilisé pour l’optimisation des hyperparamètres en testant différentes combinaisons afin de minimiser la perte de validation du modèle.

**Création de la Fonction Objective** : La fonction objective évalue différentes configurations d'hyperparamètres en entraînant le modèle et en enregistrant la perte de validation.
   
**Utilisation de Callbacks de Pruning** : Un callback spécifique a été mis en place pour stopper l’entraînement des essais qui ne montraient pas d’amélioration significative, permettant ainsi d’accélérer l’optimisation.

## Hyperparamètres des modèles utilisés pour l'entraînement après optimisation

### Hyperparamètres du modèle DenseNet 

- **Taux de croissance (« growth_rate » = 8)** : Contrôle le nombre de canaux après chaque couche dans les blocs denses.
- **Nombre de couches dans chaque bloc (« block_layers » = [4,8,12,8])** : Définit le nombre de couches de convolution 3x3 dans chaque bloc dense.
- **Nombre de classes (« num_classes » = 2)** : Correspond au nombre de sorties de la couche fully connected, soit le nombre de classes pour la classification.
- **Taux de Dropout (« dropout » = 0)** : Définit la probabilité pour chaque neurone d'être désactivé pendant l'entraînement pour réduire le risque de surapprentissage.
- **Taux d’apprentissage (« learning_rate » = 3.78 x 10⁻⁵)** : Indique la vitesse à laquelle les poids sont mis à jour pendant l’entraînement.
- **Pondération L2 (« weight_decay » = 0)** : Pénalise les grands poids en ajoutant la somme des carrés des poids à la fonction perte, réduisant le risque de surapprentissage.
- **Nombre d’époques / epoch (« max_epochs » = 100)** : Nombre maximal de passes sur l’ensemble des données d’entraînement.
- **Batch size (« batch_size » = 128)** : Nombre d’exemples d’entraînement utilisés pour calculer le gradient lors d’une passe d’entraînement.
- **Taille du Kernel (« kernel_size »)** : Dimension du noyau de convolution utilisé pendant l’opération de convolution.
- **Arrêt prématuré / Early stopping (« patience » = 8)** : Arrête l’entraînement lorsque le loss de validation reste constant ou augmente sur un nombre d'epochs définis.
- **Stride (« Stride »)** : Pas de déplacement du filtre de convolution à travers l’image d’entrée.
- **Padding (« padding »)** : Ajoute des pixels supplémentaires autour des bords de l’image avant l’application du filtre de convolution pour maintenir la taille des dimensions spatiales de la sortie.
- **Scheduler de réduction du taux d'apprentissage (« ReduceLROnPlateau »)** : Réduit le taux d'apprentissage lorsque la métrique surveillée reste constante.
- Facteur de réduction du taux d'apprentissage (« **factor** » = 0.2) : Il définit le facteur par lequel le taux d'apprentissage est réduit lorsque le scheduler est déclenché.

### Hyperparamètres du modèle U-Net 

- **Learning Rate (« learning_rate » = 3.5 x 10⁻⁴)** : Indique la vitesse à laquelle les poids sont mis à jour pendant l’entraînement.
- **Nombre de canaux des couches convolutives (« in_channels », « out_channels »)** : Définit le nombre de canaux en entrée et en sortie pour chaque couche convolutionnelle.
- **Kernel Size (« kernel_size »)** : Dimension du noyau de convolution utilisé pendant l’opération de convolution.
- **Padding (« padding »)** : Ajoute des pixels supplémentaires autour des bords de l’image avant l’application du filtre de convolution.
- **Stride (« stride »)** : Pas de déplacement du filtre de convolution à travers l’image d’entrée.
- **Nombre d’époques (« max_epochs » = 100)** : Nombre maximum de passes sur l'ensemble des données d'entraînement.
- **Batch Size (« batch_size » = 256)** : Nombre d’exemples d’entraînement utilisés pour calculer le gradient lors d’une passe d’entraînement.
- **Scheduler de réduction du taux d'apprentissage (« ReduceLROnPlateau »)** : Réduit le taux d'apprentissage lorsqu’une métrique surveillée reste constante.
- **Early Stopping (« patience » = 9)** : Interrompt l'entraînement lorsque la perte de validation ne s’améliore pas sur un nombre d'époques défini.
- Facteur de réduction du taux d'apprentissage (« **factor** » = 0.2) : Il définit le facteur par lequel le taux d'apprentissage est réduit lorsque le scheduler est déclenché.
- **Attention Kernel Size (« kernel_size » = 7)** : Taille du noyau utilisé dans le module d'attention spatiale pour capturer les caractéristiques des cartes de caractéristiques.

## Interface utilisateur pour tester les réseaux de neurones

Le projet inclut une interface utilisateur graphique (GUI) qui permet de tester facilement les réseaux de neurones pour la détection et la correction des artefacts de mouvement dans les images IRM. Cette application, développée en Python avec PyQt5, propose les fonctionnalités suivantes :

- **Chargement des Images** : Permet de charger des images IRM avec des artefacts de mouvement directement depuis votre ordinateur.
- **Détection des Artefacts** : Utilise le modèle DenseNet pour détecter la présence d'artefacts de mouvement dans les images chargées.
- **Correction des Artefacts** : Si des artefacts sont détectés, le modèle U-Net est utilisé pour corriger l'image, améliorant ainsi sa qualité.
- **Ajustement du Facteur de Correction** : Un curseur est disponible pour ajuster dynamiquement le facteur de correction appliqué aux résidus de mouvement.
- **Redimensionnement des Images** : Des champs de saisie permettent de redimensionner les images affichées selon les besoins de l'utilisateur.
- **Visualisation de la Grad-CAM** : L'interface permet également d'afficher la Grad-CAM (Gradient-weighted Class Activation Mapping), qui met en évidence les zones de l'image ayant le plus influencé la détection des artefacts par le modèle. 


