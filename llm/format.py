import os
import base64
import dataclasses
from PIL import Image
from typing import Literal, List

import google.generativeai as genai
GOOGLE_API_KEY = os.getenv(f"GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

Roles = Literal["system", "user", "assistant"]

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

@dataclasses.dataclass()
class Message:
    role: Roles
    content: str
    image_paths: List[str] = None
    
    
    def to_openai_format(self):
        if self.image_paths:
            content =  [{"type": "text", "text": self.content}]
            content += [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}}
                for image_path in self.image_paths
            ]
            return {"role": self.role, "content": content}
        else:
            return {"role": self.role, "content": self.content}
    
    
    def to_gemini_format(self):
        if self.image_paths:
            images = [genai.upload_file(path=image_path) for image_path in self.image_paths]
            inputs = images + ["\n\n", self.content]
        else:
            inputs = [self.content]
        gemini_role = {"system": "user", "assistant": "model"}.get(self.role, "user")
        return { "role": gemini_role, "parts": inputs}
    
    
    def to_llava_format(self):
        if self.image_paths:
            images = [Image.open(image_path) for image_path in self.image_paths]
            contents_list = [{"type": "image"} for image in images]
            contents_list.append({"type": "text", "text": self.content})
            return images, {
                "role": self.role,
                "content": contents_list
            }
        else:
            return {
                "role": self.role,
                "content": {"type": "text", "text": self.content}
            }

    
    def to_llama_format(self):
        return {"role": self.role, "content": self.content}
