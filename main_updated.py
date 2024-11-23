"""
Load the models
"""
from llm.api.openai_models import GPTChat
from llm.api.gemini_models import GeminiChat
from llm.api.azure_openai_models import AzureChat
from llm.api.deepinfra_models import DeepInfraChat
from llm.local.llava_model import LlavaChat
from llm.local.llama_model import LlamaChat
# mllm = DeepInfraChat("meta-llama/Llama-3.2-11B-Vision-Instruct")
# llm = AzureChat("gpt-4o-mini")

# llm = GeminiChat("gemini-1.5-pro")
# mllm = GeminiChat("gemini-1.5-pro")

mllm = LlavaChat("llava-hf/llava-v1.6-mistral-7b-hf")
llm = LlamaChat("meta-llama/Meta-Llama-3.1-8B-Instruct")

"""
Load the dataset through customized json file
"""
from dataset import *
dataset = ImageDataset(llm)   # the model here used for "ENTITY EXTRACTION" !!!

#%%
prefix = "mmvet_llavanext_"
#%%
""" Entity Extraction """
dataset.load("load/mmvet_small.json")
dataset.new_generate_caption(mllm, prefix + "1_caption_generation")    # model used for "CAPTION GENERATION" !!!
#%%
""" Question Decomposition """
dataset.load(f"save/{prefix}1_caption_generation.json")
dataset.generate_subquestion_overall(llm, prefix + "3_test_subquestion")
#%%
"""
SubQuestion Answering (original image + cropped image)
"""
dataset.load(f"save/{prefix}3_test_subquestion.json")
dataset.subquestion_answering_overall(mllm, prefix + "4_test_answer")
#%%
"""
Summarization (Summarize the subquestion-answer pairs)
    - if no subquestion, just returns empty string info ""
"""
dataset.load(f"save/{prefix}4_test_answer.json")
dataset.subquestion_summarization_overall(llm, prefix + "5_test_summarize")

#%%
"""
Final Question Answering
"""
dataset.load(f"save/{prefix}5_test_summarize.json")
dataset.question_answering(mllm, prefix + "6_test_answer")

#%%
import json

with open(f"save/{prefix}6_test_answer.json", "r") as f:
    data = json.load(f)

with open("results.json", "w") as output_f:
    results = {sol["id"]: sol["summarization"] for sol in data}
    json.dump(results, output_f, indent=4)
