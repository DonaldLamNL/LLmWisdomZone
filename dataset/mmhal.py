import os
from tqdm.auto import tqdm
from datasets import load_dataset

from utils.global_functions import *
from dataset.base import ImageDataset

class MMHal(ImageDataset):
    def __init__(self, dataset_name: str, cache_dir: str, output_path: str):
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.output_path = output_path
        super().__init__()
    
    
    def load_dataset(self):
        return load_dataset(
            self.dataset_name,
            "default",
            cache_dir=self.cache_dir
        )
    
    
    def process(self):
        for entry in tqdm(self.ds["test"]):
            image_id = entry["image_id"]
            jpg_path = f"/content/mmhal_data/mmhal/{image_id}.jpg"
            png_path = f"/content/mmhal_data/mmhal/{image_id}.png"
            image_path = jpg_path if os.path.exists(jpg_path) else (png_path if os.path.exists(png_path) else "Image not found")
            
            self.data.append({
                "question_type": entry["question_type"],
                "question_topic": entry["question_topic"],
                "image_content": entry["image_content"],
                "question": entry["question"],
                "gt_answer": entry["gt_answer"],
                "image_id": image_id,
                "image_path": image_path, 
            })