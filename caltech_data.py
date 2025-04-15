import torch
from torchvision.transforms import v2
from torch.utils.data import random_split,DataLoader
import os
import urllib.request
import tarfile
from typing import Optional, Callable, Any
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader


data_dir = os.path.join(os.getcwd(),"data")

PATH_TO_CALTECH256 = os.path.join(data_dir,"caltech256")

# PATH_TO_CALTECH256 = "/mnt/769EC2439EC1FB9D/vsc_projs/caltech256"



class CustomCaltech256(ImageFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        download: bool = False,
        custom_url: Optional[str] = None,
        filename: str = "256_ObjectCategories.tar",
    ):
        self.root = os.path.expanduser(root)
        self.custom_url = custom_url or (
            "https://data.caltech.edu/records/nyy15-4j048/files/256_ObjectCategories.tar?download=1"
        )
        self.filename = filename
        self.filepath = os.path.join(self.root, self.filename)
        self.data_folder = os.path.join(self.root, "256_ObjectCategories")

        if download:
            self._download()

        super().__init__(
            root=self.data_folder,
            transform=transform,
            target_transform=target_transform,
            loader=loader
        )

    def _download(self):
        if os.path.isdir(self.data_folder):
            print("✅ Caltech-256 already extracted.")
            return

        os.makedirs(self.root, exist_ok=True)

        if not os.path.isfile(self.filepath):
            print("⬇️ Downloading Caltech-256...")

            def progress_hook(t):
                last_b = [0]
                def update_to(block_num=1, block_size=1, total_size=None):
                    if total_size is not None:
                        t.total = total_size
                    downloaded = block_num * block_size
                    t.update(downloaded - last_b[0])
                    last_b[0] = downloaded
                return update_to

            with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=self.filename) as t:
                urllib.request.urlretrieve(self.custom_url, self.filepath, reporthook=progress_hook(t))

            print("✅ Download complete.")

        print("📦 Extracting Caltech-256...")
        with tarfile.open(self.filepath, "r") as tar:
            tar.extractall(path=self.root)
        print("✅ Extraction complete.")


transforms = v2.Compose([
    v2.PILToTensor(),
    v2.ToDtype(torch.float32,scale=True),
    v2.Resize(256),
    v2.CenterCrop(224),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),    #imagenet norm values...

    # augments if to be added
    v2.AutoAugment()

])

caltech256 = CustomCaltech256(
    root=PATH_TO_CALTECH256,
    transform=transforms,
    download=True,
)
train_data,val_data = random_split(caltech256,[27607,3000])     # ~90/10

def get_caltech_train_loader(batch_size,shuffle=True,num_workers=4):
    return DataLoader(train_data,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory=True)

def get_caltech_val_loader(batch_size,shuffle=True,num_workers=4):
    return DataLoader(val_data,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory=True)


