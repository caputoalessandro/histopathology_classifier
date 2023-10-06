import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image, ImageFile
from torchvision.transforms.functional import adjust_contrast

ImageFile.LOAD_TRUNCATED_IMAGES = True


class EIDOSTilesDataset(Dataset):
    def __init__(self, annotations_file, img_dir, num_classes, transform=None):
        self.img_annotations = pd.read_table(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.img_annotations)

    def __getitem__(self, idx):
        tile_path = str(self.img_dir / self.img_annotations["tile_name"][idx])
        tile = Image.open(tile_path).convert("RGB")
        target = self.convert__label(self.img_annotations["tile_label"][idx])
        if self.transform:
            tile = self.transform(tile)

        return tile, target

    def get_tile_path(self, idx):
        return str(self.img_dir / self.img_annotations["tile_name"][idx])

    def convert__label(self, target):
        conversion = (
            {
                "I": 0,
                "II": 0,
                "III": 1,
                "IV": 1,
            }
            if self.num_classes == 2
            else {
                "I": 0,
                "II": 1,
                "III": 2,
                "IV": 3,
            }
        )
        return torch.tensor(conversion[target])


class EIDOSWSIdataset(Dataset):
    def __init__(self, annotations_file, img_dir):
        self.img_annotations = annotations_file
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_annotations)

    def __getitem__(self, idx):
        wsi_path = self.img_dir / self.img_annotations["filename"][idx]
        patient_info = self.img_annotations.iloc[idx]
        target = patient_info["Stage"]
        return wsi_path, target, patient_info
