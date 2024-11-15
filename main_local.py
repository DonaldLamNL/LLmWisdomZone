"""
Load the models
"""
from llm import GPTChat
from llm.local.llava_model import LlavaChat
from llm.local.llama_model import LlamaChat
gpt4omini = GPTChat("gpt-4o-mini")
llava = LlavaChat("llava-hf/llava-v1.6-mistral-7b-hf")
llama = LlamaChat("meta-llama/Meta-Llama-3.1-8B-Instruct")




"""
Load the dataset through customized json file
"""
from dataset import *
dataset = ImageDataset(llama)   # the model here used for "ENTITY EXTRACTION" !!!
dataset.load("load/mmvet.json")


"""
Entity Extraction
"""
dataset.entity_extraction(llava)    # model used for "CAPTION GENERATION" !!!
dataset.save_json("1_entity_extraction")


"""
Object Detection
"""
dataset.object_detection()
dataset.save_json("2_object_detection")


"""
Question Decomposition
"""
dataset.generate_subquestion(llama)
dataset.save_json("3_test_subquestion")


"""
SubQuestion Answering (original image + cropped image)
"""
dataset.subquestion_answering(llava)
dataset.save_json("4_test_answer")


"""
Summarization (Summarize the subquestion-answer pairs)
    - if no subquestion, just returns empty string info ""
"""
dataset.subquestion_summarization(llama)
dataset.save_json("5_test_summarize")


"""
Final Question Answering
"""
dataset.question_answering(llava)
dataset.save_json("6_test_answer")
