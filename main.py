"""
Load the models
"""
from llm import GPTChat
from llm.local.llava_model import LlavaChat
from llm.local.llama_model import LlamaChat
gpt4omini = GPTChat("gpt-4o-mini")
llava = LlavaChat()
llama = LlamaChat()


"""
Load the dataset through customized json file
"""
from dataset import *
dataset = ImageDataset(gpt4omini)   # the model here used for "ENTITY EXTRACTION" !!!
dataset.load("load/mmvet.json")


"""
Entity Extraction
"""
dataset.entity_extraction(gpt4omini)    # model used for "CAPTION GENERATION" !!!
dataset.save_json("1_entity_extraction")


"""
Object Detection
"""
dataset.object_detection()
dataset.save_json("2_object_detection")


"""
Question Decomposition
"""
dataset.generate_subquestion(gpt4omini)
dataset.save_json("3_test_subquestion")


"""
SubQuestion Answering (original image + cropped image)
"""
dataset.subquestion_answering(gpt4omini)
dataset.save_json("4_test_answer")


"""
Summarization (Summarize the subquestion-answer pairs)
    - if no subquestion, just returns empty string info ""
"""
dataset.subquestion_summarization(gpt4omini)
dataset.save_json("5_test_summarize")


"""
Final Question Answering
"""
dataset.question_answering(gpt4omini)
dataset.save_json("6_test_answer")


from debator.base import Debate
debator = Debate(llm=gpt4omini, mllm=gpt4omini)
dataset.debate(debator)
dataset.save_json("7_debater")
