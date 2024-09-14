# Détection et Correction des Artefacts de Mouvement sur les Images d'IRM Cérébrales

## Description du Projet

Ce projet vise à développer des méthodes basées sur l'intelligence artificielle pour détecter et corriger les artefacts de mouvement sur les images d'Imagerie par Résonance Magnétique (IRM) cérébrales. Les artefacts de mouvement, causés par des déplacements volontaires ou involontaires des patients lors des acquisitions IRM, altèrent la qualité des images et compliquent le diagnostic médical. En s'appuyant sur des réseaux de neurones profonds tels que DenseNet pour la détection et U-Net pour la correction, ce projet propose une solution pour automatiser le processus de correction des images, améliorant ainsi leur qualité et facilitant l'interprétation clinique.

## Objectifs du Projet

- Étudier les artefacts de mouvement et leur impact sur les images IRM cérébrales.
- Développer des algorithmes de détection des artefacts de mouvement à l'aide de réseaux de neurones convolutifs.
- Concevoir des modèles pour corriger ces artefacts, en restaurant les images affectées à partir de données simulées et réelles.
- Évaluer les performances des modèles à travers des métriques de qualité d'image telles que l'indice de similarité structurelle (SSIM).

## Méthodologie

1. **Étude des Artefacts de Mouvement** : Analyse des types d'artefacts causés par les mouvements et de leur impact sur les images IRM.
2. **Développement de Modèles** : Utilisation de DenseNet pour détecter les artefacts et de U-Net pour corriger les images dégradées. Les modèles sont entraînés sur des jeux de données comprenant des images IRM avec et sans artefacts.
3. **Entraînement et Évaluation** : Les réseaux de neurones sont entraînés sur des serveurs équipés de GPU pour maximiser l'efficacité du traitement. Les performances sont évaluées à l'aide de métriques comme le SSIM.
4. **Optimisation** : Ajustement des hyperparamètres via des techniques d'optimisation bayésienne pour améliorer la précision et la fiabilité des modèles.

## Technologies Utilisées

- **Python** : Langage principal pour le développement des algorithmes.
- **PyTorch** : Framework utilisé pour l'implémentation des réseaux de neurones.
- **Optuna** : Bibliothèque pour l'optimisation des hyperparamètres.



