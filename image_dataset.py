from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from glob import glob
import random
import scipy.io
from torchvision.io import read_image


class ImageDataset(Dataset):
    """
    Dataset class for the image patches
    """

    def __init__(self, dataset_path, mat_file, key, patch_size, transform=None):
        self.path = dataset_path
        self.patch_size = patch_size
        self.transform = transform

        # Load ground truth illumination values
        illum_mat = scipy.io.loadmat(mat_file, squeeze_me=True, struct_as_record=False)
        self.ground_truth_illum = illum_mat[key]

        self.file_list = glob(self.path + '*.png')

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        # Read image
        image = read_image(img_path)
        # Convert to RGB
        image = transforms.Grayscale(num_output_channels=3)(image)

        # Generate a random patch from the image
        patch_r, patch_c = self.patch_size
        start_row = random.randint(0, image.shape[0] - patch_r)
        start_col = random.randint(0, image.shape[1] - patch_c)
        patch = image[start_row:start_row+patch_r, start_col:start_col+patch_c]

        # Apply transformations
        if self.transform:
            patch = self.transform(patch)

        # Get the corresponding ground truth illumination
        index = int(img_path.replace(self.path, '').replace('.png', '')) - 1
        ground_truth = self.ground_truth_illum[index]

        return patch, ground_truth