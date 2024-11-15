import os
from tqdm.auto import tqdm
from datasets import load_dataset

from utils.global_functions import *
from dataset.base import ImageDataset

class MMVet(ImageDataset):
    def __init__(self, model, dataset_name: str, cache_dir: str, output_path: str):
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.output_path = output_path
        self.ds = self.load_dataset()
        super().__init__(model)
    
    
    def load_dataset(self):
        return load_dataset(
            self.dataset_name,
            "default",
            cache_dir=self.cache_dir
        )
    
    
    def process(self):
        for entry in tqdm(self.ds["test"]):
            image_id = entry["id"]
            jpg_path = f"/content/mmvet_data/mmvet/{image_id}.jpg"
            png_path = f"/content/mmvet_data/mmvet/{image_id}.png"
            image_path = jpg_path if os.path.exists(jpg_path) else (png_path if os.path.exists(png_path) else "Image not found")
            
            self.data.append({
                "id": image_id,
                "image": entry["image"],
                "image_path": image_path,
                "question": entry["question"],
                "ground_truth": entry["answer"],
            })