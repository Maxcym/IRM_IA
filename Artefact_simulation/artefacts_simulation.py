import csv
import os
import numpy as np
import cv2
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt


class MotionArtifactSimulator:
    """
    Classe pour simuler des artefacts de mouvement dans les images médicales en appliquant des décalages de phase dans l'espace k.

    Attributs :
        input_folder (str) : Dossier contenant les images d'entrée.
        output_folder (str) : Dossier où les images corrompues seront enregistrées.
        num_images (int) : Nombre d'images corrompues à générer par image d'entrée.
        csv_output_folder (str) : Dossier où les fichiers CSV contenant les informations de décalage de phase seront enregistrés.
        current_image_name (str) : Nom de l'image actuellement traitée.
    """

    def __init__(self, input_folder, output_folder, num_images, csv_output_folder):
        """
        Initialise les paramètres pour la simulation d'artefacts de mouvement.

        Paramètres :
            input_folder (str) : Dossier contenant les images d'entrée.
            output_folder (str) : Dossier où les images corrompues seront enregistrées.
            num_images (int) : Nombre d'images corrompues à générer par image d'entrée.
            csv_output_folder (str) : Dossier où les fichiers CSV contenant les informations de décalage de phase seront enregistrés.
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.num_images = num_images
        self.csv_output_folder = csv_output_folder
        self.current_image_name = ""

    def apply_phase_shift(self, k_space, num_lines, image_number, region):
        """
        Applique des décalages de phase linéaires aléatoires sur des régions spécifiques de l'espace k et enregistre ces décalages.

        Paramètres :
            k_space (ndarray) : Représentation de l'image dans l'espace k.
            num_lines (int) : Nombre de lignes de l'espace k sur lesquelles appliquer les décalages de phase.
            image_number (int) : Numéro de l'image en cours de traitement.
            region (str) : Région de l'espace k à affecter ('center' ou 'edges').

        Retourne :
            tuple : Contient l'espace k modifié avec les décalages de phase appliqués,
                    le tableau des décalages de phase, et le pourcentage de lignes affectées.
        """
        rows, cols = k_space.shape
        phase_shifts = np.zeros((rows, cols))
        csv_file_path = os.path.join(self.csv_output_folder, f'phase_shifts_{region}.csv')

        # Définition de la région affectée dans l'espace k en fonction de la région choisie
        if region == 'center':
            start, end = int(rows * 0.45), int(rows * 0.55)
        elif region == 'edges':
            start, end = int(rows * 0.425), int(rows * 0.575)

        file_is_empty = not os.path.isfile(csv_file_path) or os.path.getsize(csv_file_path) == 0

        affected_lines = set()
        with open(csv_file_path, 'a', newline='') as csvfile:
            fieldnames = ['Image Name', 'Image Number', 'Line', 'Phase Shift Angle']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Écrit l'en-tête si le fichier est vide
            if file_is_empty:
                writer.writeheader()

            # Applique des décalages de phase à des lignes aléatoires dans la région spécifiée
            for _ in range(num_lines):
                line = np.random.randint(0, rows)
                if region == 'edges' and start <= line < end:
                    line = (line + (end - start)) % rows
                elif region == 'center':
                    line = np.random.randint(start, end)

                affected_lines.add(line)

                phase_shift = np.exp(-2j * np.pi * np.random.rand() * np.arange(cols) / cols)
                k_space[line, :] *= phase_shift
                phase_shifts[line, :] = np.angle(phase_shift)

                # Enregistre les informations des décalages de phase
                writer.writerow({
                    'Image Name': self.current_image_name,
                    'Image Number': image_number,
                    'Line': line,
                    'Phase Shift Angle': phase_shifts[line, :].tolist()
                })

        affected_percentage = (len(affected_lines) / rows) * 100
        return k_space, phase_shifts, affected_percentage

    def save_phase_shifts_image(self, phase_shifts, image_name, region, image_number, affected_percentage):
        """
        Sauvegarde les décalages de phase sous forme d'image pour la visualisation.

        Paramètres :
            phase_shifts (ndarray) : Les décalages de phase à sauvegarder en tant qu'image.
            image_name (str) : Nom de base de l'image.
            region (str) : Région de l'espace k ('center' ou 'edges').
            image_number (int) : Numéro de l'image en cours de traitement.
            affected_percentage (float) : Pourcentage de lignes affectées par les décalages de phase.
        """
        plt.figure(figsize=(10, 4))
        plt.imshow(phase_shifts, cmap='jet', aspect='auto')
        plt.colorbar()
        plt.title(
            f'Décalages de Phase ({region.capitalize()}) - Image {image_number}\n{affected_percentage:.2f}% de lignes affectées')
        plt.xlabel('Encodage de Fréquence')
        plt.ylabel('Encodage de Phase')

        # Sauvegarde de l'image des décalages de phase
        phase_shifts_image_path = os.path.join("../Phase_shifts_graph", f"{os.path.splitext(image_name)[0]}_phase_shifts_{region}_{image_number}.png")
        plt.savefig(phase_shifts_image_path)
        plt.close()

    def simulate_artifacts(self, region):
        """
        Simule des artefacts de mouvement en appliquant des décalages de phase aléatoires sur des régions spécifiques de l'espace k.

        Paramètres :
            region (str) : La région de l'espace k à affecter ('center' ou 'edges').
        """
        # Parcourt toutes les images dans le dossier input_folder
        for image_name in os.listdir(self.input_folder):
            image_path = os.path.join(self.input_folder, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            self.current_image_name = image_name

            # Génère un artefact par image
            for i in range(1, self.num_images + 1):
                k_space = fftshift(fft2(image))
                if region == 'center':
                    k_space_modified, phase_shifts, affected_percentage = self.apply_phase_shift(np.copy(k_space), 30, i, region)
                    corrupted_image_path = os.path.join(self.output_folder, f"{os.path.splitext(image_name)[0]}_artefact_bf_{i}.png")
                elif region == 'edges':
                    k_space_modified, phase_shifts, affected_percentage = self.apply_phase_shift(np.copy(k_space), 70, i, region)
                    corrupted_image_path = os.path.join(self.output_folder, f"{os.path.splitext(image_name)[0]}_artefact_hf_{i}.png")

                corrupted_image = np.abs(ifft2(ifftshift(k_space_modified)))
                cv2.imwrite(corrupted_image_path, corrupted_image)

                # Sauvegarde des résidus et des images de décalage de phase
                residuals = image - corrupted_image
                residuals_normalized = cv2.normalize(residuals, None, 0, 255, cv2.NORM_MINMAX)
                residuals_image_path = os.path.join("../Residus_mov", f"{os.path.splitext(image_name)[0]}_residuals_{region}_{i}.png")
                cv2.imwrite(residuals_image_path, residuals_normalized)

                # Enregistre l'image des décalages de phase
                self.save_phase_shifts_image(phase_shifts, image_name, region, i, affected_percentage)

                # Appel de plot_results pour afficher les résultats
                self.plot_results(image, corrupted_image, phase_shifts, i)

    def plot_results(self, original_image, corrupted_image, phase_shifts, image_number):
        """
        Affiche l'image originale, l'image corrompue, les décalages de phase et les résidus de mouvement.

        Paramètres :
            original_image (ndarray) : L'image originale.
            corrupted_image (ndarray) : L'image avec des artefacts de mouvement.
            phase_shifts (ndarray) : Les décalages de phase appliqués dans l'espace k.
            image_number (int) : Numéro de l'image en cours de traitement.
        """
        residuals = original_image - corrupted_image

        # Affichage des images originale, corrompue, des décalages de phase et des résidus
        f, s = plt.subplots(1, 4, figsize=(20, 5))
        s[0].imshow(original_image, cmap='gray')
        s[0].axis('off')
        s[0].set_title('Original')

        s[1].imshow(corrupted_image, cmap='gray')
        s[1].axis('off')
        s[1].set_title(f'Corrompue (Image {image_number})')

        s[2].imshow(phase_shifts, cmap='jet', aspect='auto')
        s[2].set_title('Décalages de Phase')
        s[2].set_xlabel('Encodage de Fréquence')
        s[2].set_ylabel('Encodage de Phase')

        s[3].imshow(residuals, cmap='gray')
        s[3].axis('off')
        s[3].set_title('Résidus de Mouvement')

        plt.show()


# Exemple d'utilisation
input_folder = "C:/Users/maxim/Desktop/Dataset_test/Class_0"
output_folder = "C:/Users/maxim/Desktop/Output_Slices"
csv_output_folder = "../Data_shift"
num_images = 1

# Assurez-vous que le dossier de sortie CSV existe
os.makedirs(csv_output_folder, exist_ok=True)

# Instancie et exécute le simulateur
simulator = MotionArtifactSimulator(input_folder, output_folder, num_images, csv_output_folder)
simulator.simulate_artifacts('edges')
# Décommenter la ligne suivante pour simuler des artefacts dans la région centrale
# simulator.simulate_artifacts('center')

