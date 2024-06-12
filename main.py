import os
import nibabel as nib
import numpy as np
from scipy.stats import ttest_ind
from skimage import filters, morphology

def load_mri_images(folder_path):
    """
    Ładuje obrazy MRI z podanego folderu.

    Args:
        folder_path (str): Ścieżka do folderu zawierającego obrazy MRI w formacie .nii.

    Returns:
        list: Lista krotek, gdzie każda krotka zawiera nazwę pliku i załadowany obraz MRI.
    """
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".nii"):
            file_path = os.path.join(folder_path, filename)
            image = nib.load(file_path)
            images.append((filename, image))
    return images

def segment_hippocampus(image_data):
    """
    Segmentuje hipokamp z danych obrazu MRI.

    Args:
        image_data (numpy.ndarray): Dane obrazu MRI.

    Returns:
        numpy.ndarray: Dane obrazu MRI z zaznaczonym hipokampem.
    """
    middle_slice = image_data[:, :, image_data.shape[2] // 2]

    # Użycie metody Otsu do progowania
    threshold_value = filters.threshold_otsu(middle_slice)
    binary_image = middle_slice > threshold_value

    # Usuwanie małych obiektów i małych dziur
    cleaned_image = morphology.remove_small_objects(binary_image, min_size=64)
    cleaned_image = morphology.remove_small_holes(cleaned_image, area_threshold=64)

    # Tworzenie nowej tablicy do przechowywania segmentacji
    segmented_data = np.zeros_like(image_data)
    segmented_data[:, :, image_data.shape[2] // 2] = cleaned_image

    return segmented_data

def calculate_volume(segmented_data):
    """
    Oblicza objętość hipokampu na podstawie segmentacji.

    Args:
        segmented_data (numpy.ndarray): Dane obrazu MRI z zaznaczonym hipokampem.

    Returns:
        float: Objętość hipokampu.
    """
    # Obliczenie objętości woksela
    voxel_volume = np.prod(segmented_data.header.get_zooms())
    hippocampus_volume = np.sum(segmented_data.get_fdata() > 0) * voxel_volume
    return hippocampus_volume

def analyze_volumes(folder_path):
    """
    Analizuje objętości hipokampu w obrazach MRI z podanego folderu.

    Args:
        folder_path (str): Ścieżka do folderu zawierającego obrazy MRI.

    Returns:
        list: Lista objętości hipokampu dla każdego obrazu.
    """
    volumes = []
    images = load_mri_images(folder_path)
    for filename, image in images:
        image_data = image.get_fdata()
        segmented_data = segment_hippocampus(image_data)
        volume = calculate_volume(segmented_data)
        volumes.append(volume)
    return volumes

def main(treated_baseline_path, treated_6months_path, treated_12months_path,
         control_baseline_path, control_6months_path, control_12months_path):
    """
    Główna funkcja analizująca zmiany objętości hipokampu w czasie dla grup leczonych i kontrolnych.

    Args:
        treated_baseline_path (str): Ścieżka do folderu z obrazami MRI pacjentów leczonych (stan wyjściowy).
        treated_6months_path (str): Ścieżka do folderu z obrazami MRI pacjentów leczonych (po 6 miesiącach).
        treated_12months_path (str): Ścieżka do folderu z obrazami MRI pacjentów leczonych (po 12 miesiącach).
        control_baseline_path (str): Ścieżka do folderu z obrazami MRI pacjentów kontrolnych (stan wyjściowy).
        control_6months_path (str): Ścieżka do folderu z obrazami MRI pacjentów kontrolnych (po 6 miesiącach).
        control_12months_path (str): Ścieżka do folderu z obrazami MRI pacjentów kontrolnych (po 12 miesiącach).
    """
    # Analiza objętości hipokampu dla pacjentów leczonych
    treated_volumes_baseline = analyze_volumes(treated_baseline_path)
    treated_volumes_6months = analyze_volumes(treated_6months_path)
    treated_volumes_12months = analyze_volumes(treated_12months_path)

    # Analiza objętości hipokampu dla pacjentów kontrolnych
    control_volumes_baseline = analyze_volumes(control_baseline_path)
    control_volumes_6months = analyze_volumes(control_6months_path)
    control_volumes_12months = analyze_volumes(control_12months_path)

    # Obliczenie zmian objętości hipokampu dla pacjentów leczonych
    treated_changes_6m = [treated_volumes_6months[i] - treated_volumes_baseline[i] for i in
                          range(len(treated_volumes_baseline))]
    treated_changes_12m = [treated_volumes_12months[i]
