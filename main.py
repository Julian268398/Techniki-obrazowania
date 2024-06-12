import os
import nibabel as nib
import numpy as np
from scipy.stats import ttest_ind
from skimage import filters, morphology


def load_mri_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".nii"):
            file_path = os.path.join(folder_path, filename)
            image = nib.load(file_path)
            images.append((filename, image))
    return images


def segment_hippocampus(image_data):
    middle_slice = image_data[:, :, image_data.shape[2] // 2]

    threshold_value = filters.threshold_otsu(middle_slice)
    binary_image = middle_slice > threshold_value

    cleaned_image = morphology.remove_small_objects(binary_image, min_size=64)
    cleaned_image = morphology.remove_small_holes(cleaned_image, area_threshold=64)

    segmented_data = np.zeros_like(image_data)
    segmented_data[:, :, image_data.shape[2] // 2] = cleaned_image

    return segmented_data


def calculate_volume(segmented_data):
    voxel_volume = np.prod(segmented_data.header.get_zooms())
    hippocampus_volume = np.sum(segmented_data.get_fdata() > 0) * voxel_volume
    return hippocampus_volume


def analyze_volumes(folder_path):
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
    treated_volumes_baseline = analyze_volumes(treated_baseline_path)
    treated_volumes_6months = analyze_volumes(treated_6months_path)
    treated_volumes_12months = analyze_volumes(treated_12months_path)

    control_volumes_baseline = analyze_volumes(control_baseline_path)
    control_volumes_6months = analyze_volumes(control_6months_path)
    control_volumes_12months = analyze_volumes(control_12months_path)

    treated_changes_6m = [treated_volumes_6months[i] - treated_volumes_baseline[i] for i in
                          range(len(treated_volumes_baseline))]
    treated_changes_12m = [treated_volumes_12months[i] - treated_volumes_baseline[i] for i in
                           range(len(treated_volumes_baseline))]

    control_changes_6m = [control_volumes_6months[i] - control_volumes_baseline[i] for i in
                          range(len(control_volumes_baseline))]
    control_changes_12m = [control_volumes_12months[i] - control_volumes_baseline[i] for i in
                           range(len(control_volumes_baseline))]

    t_stat_6m, p_value_6m = ttest_ind(treated_changes_6m, control_changes_6m)
    t_stat_12m, p_value_12m = ttest_ind(treated_changes_12m, control_changes_12m)

    print(f"6 miesięcy: t={t_stat_6m}, p={p_value_6m}")
    print(f"12 miesięcy: t={t_stat_12m}, p={p_value_12m}")


if __name__ == "__main__":
    treated_baseline_path = "/path/to/treated/baseline/"
    treated_6months_path = "/path/to/treated/6months/"
    treated_12months_path = "/path/to/treated/12months/"

    control_baseline_path = "/path/to/control/baseline/"
    control_6months_path = "/path/to/control/6months/"
    control_12months_path = "/path/to/control/12months/"

    main(treated_baseline_path, treated_6months_path, treated_12months_path,
         control_baseline_path, control_6months_path, control_12months_path)
