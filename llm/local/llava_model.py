from llm.base import *
from llm.format import *
from utils.global_functions import *

import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

class LlavaChat(LLMChat):
    def __init__(self, model_name="llava-hf/llava-v1.6-mistral-7b-hf", device="cuda"):
        # Initialize the processor and model with the specified model name and device
        self.processor = LlavaNextProcessor.from_pretrained(model_name)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
        self.device = device
        self.model.to(self.device)
    
    
    def chat(self, messages: List[Message]) -> Union[List[str], str]:
        if messages[0].role == 'user':
            images, conversation = messages[0].to_llava_format()
        else:
            images, conversation = messages[1].to_llava_format()
        
        # Apply the chat template and process the inputs
        prompt = self.processor.apply_chat_template([conversation], add_generation_prompt=True)
        inputs = self.processor(images=images, text=prompt, return_tensors="pt").to(self.device)
        
        # Generate the output autoregressively
        output = self.model.generate(**inputs, temperature=1, max_new_tokens=512)
        
        # Decode the output and return the result
        result = self.processor.decode(output[0], skip_special_tokens=True).split('[/INST]')[-1].strip()
        return result
