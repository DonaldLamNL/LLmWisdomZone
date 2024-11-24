import os
import json
import wget
from tqdm.auto import tqdm

from llm import LLMChat
from utils.global_functions import *

from dataset.entity_extraction import EntityExtractor
# from object_detection import ObjectDetection

class ImageDataset():
    def __init__(self, llm_model:LLMChat,
                 object_detection_model_path='yolov8s-world.pt'):
        self.data = []
        # self.object_detector = ObjectDetection(object_detection_model_path)
        self.entity_extractor = EntityExtractor(llm_model)
    
    
    def load_dataset(self):
        pass
    
    
    def load(self, filepath:str):
        with open(filepath, "r") as file:
            self.data = json.load(file)
    
    
    def entity_extraction(self, caption_model, save) -> None:
        for entry in tqdm(self.data, desc="Processing entries for entity extraction"):
            image_path = entry["image_path"]
            question = entry["question"]
            caption = generate_caption(caption_model, question, [image_path])
            entities = self.entity_extractor.extract_entities(question, caption)
            entities = list(set(entities))
            entry["caption"] = caption
            entry["objects"] = {key: {} for key in entities}
            self.save_json(save)
    
    
    def new_generate_caption(self, caption_model:LLMChat, savefile:str) -> None:
        for entry in tqdm(self.data, desc="Processing entries for entity extraction and caption generation"):
            image_path = entry["image_path"]
            question = entry["question"]
            caption = generate_caption(caption_model, question, [image_path])
            entry["caption"] = caption
            entry["objects"] = {}
            self.save_json(savefile)
            
    
    
    def object_detection(self) -> None:
        for entry in tqdm(self.data, desc="Processing entries for object detection"):
            image_path = entry["image_path"]
            objects = entry["objects"]
            target_entities = list(objects.keys())
            entity_boxes = self.object_detector.predict(image_path, target_entities)
            output_paths_cropped_dict = {entity: [] for entity in target_entities}
            output_paths_box_dict = {entity: [] for entity in target_entities}

            parent_folder = os.path.dirname(os.path.dirname(image_path))

            # Define the path for box_image_folder
            box_image_folder_path = os.path.join(parent_folder, "box_images")

            # Check if the folder exists, and create it if it doesn't
            if not os.path.exists(box_image_folder_path):
                os.makedirs(box_image_folder_path)
                print(f"Created folder: {box_image_folder_path}")


            # Update the objects in the JSON with generated results
            for entity in target_entities:
                count = 1  # To track the number of objects detected for each entity
                output_paths_cropped = []
                output_paths_box = []
                entry["objects"][entity] = []

                # Create output paths and update JSON objects
                for box in entity_boxes.get(entity, []):
                    renamed_obj = f"{entity}_{count}"
                    # Generate the output path
                    cropped_image_path = os.path.join(box_image_folder_path,
                                                      f'{os.path.splitext(os.path.basename(image_path))[0]}_{renamed_obj}.jpg')
                    box_image_path = os.path.join(box_image_folder_path,
                                                  f'{os.path.splitext(os.path.basename(image_path))[0]}_{renamed_obj}_with_box.jpg')
                    output_paths_cropped.append(cropped_image_path)
                    output_paths_box.append(box_image_path)

                    # Update the JSON structure with new data
                    new_obj = {
                        "renamed_obj": renamed_obj,
                        "bounding_box": box,
                        "image_patch_path": cropped_image_path,
                        "image_with_box_path": box_image_path,
                        "subquestions": []  # Initialize with an empty list or populate as needed
                    }


                    entry["objects"][entity].append(new_obj)


                    count += 1

                # Create the output directory if it doesn't exist
                for path in output_paths_cropped:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                for path in output_paths_box:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                output_paths_cropped_dict[entity] = output_paths_cropped
                output_paths_box_dict[entity] = output_paths_box

            # Crop and save images based on predicted bounding boxes
            self.object_detector.crop_and_save_images(image_path, entity_boxes=entity_boxes,
                                          output_paths_cropped=output_paths_cropped_dict,
                                          output_paths_box=output_paths_box_dict)

    
    def generate_subquestion(self, model:LLMChat) -> None:
        for entry in tqdm(self.data, desc="Generating subquestions"):
            question = entry["question"]
            caption = entry["caption"]
            for entity in entry["objects"].keys():
                if len(entry["objects"][entity]) == 0: continue
                subquestions = subquestion_generator(entity, question, caption, model)
                for item in entry["objects"][entity]:
                    item["subquestions"] = subquestions
    
    
    def new_subquestion_generating(self, model:LLMChat, savefile:str) -> None:
        for entry in tqdm(self.data, desc="Generating subquestions"):
            question = entry["question"]
            caption = entry["caption"]
            entities = entry["objects"]
            subquestions = subquestion_generator(entities, question, caption, model)
            entry["subquestions"] = subquestions
            self.save_json(savefile)

    def generate_subquestion_overall(self, model:LLMChat, savefile:str) -> None:
        for entry in tqdm(self.data, desc="Generating subquestions"):
            question = entry["question"]
            caption = entry["caption"]
            subquestions = subquestion_generator_overall(question, caption, model)
            entry["subquestions_all"] = subquestions
        self.save_json(savefile)

    def subquestion_answering_overall(self, model: LLMChat, savefile:str) -> None:
        for entry in tqdm(self.data, desc="Generating subquestions answers"):
            for subquestion in entry["subquestions_all"]:
                """Load image"""
                image_path = entry["image_path"]
                target_question = entry["question"]
                image_paths = [image_path]
                question = subquestion["question"]
                caption = entry["caption"]
                answer = subquestion_answering_overall(image_paths, caption, question, target_question, model)
                subquestion["answer"] = answer
        self.save_json(savefile)

            
            
    def new_subquestion_answering(self, model:LLMChat, savefile:str) -> None:
        for entry in tqdm(self.data, desc="Answering subquestions"):
            if entry["subquestions"] is None:
                entry["subquestions"] = []
                continue
            for subquestion in tqdm(entry["subquestions"]):
                caption = entry["caption"]
                question = subquestion["question"]
                image_path = entry["image_path"]
                entities = entry["objects"]
                answer = subquestion_answering([image_path], caption, question, entities, model)
                subquestion["answer"] = answer
                self.save_json(savefile)
    
    
    def subquestion_answering(self, model:LLMChat, two_images=False) -> None:
        for entry in tqdm(self.data, desc="Generating subquestions"):
            for name, entities in entry["objects"].items():
                if len(entities) == 0: continue
                for sample in entities:
                    for subquestion in sample["subquestions"]:
                        
                        """Load image"""
                        image_path = entry["image_path"]
                        crop_image_path = sample["image_patch_path"]
                        
                        # Modify image loading here!!!
                        if two_images:
                            image_paths = [image_path, crop_image_path]
                        else:
                            image_paths = [crop_image_path]
                        
                        question = subquestion["question"]
                        
                        caption = entry["caption"]
                        
                        entity = sample["renamed_obj"]
                                                
                        answer = subquestion_answering(image_paths, caption, question, entity, model)
                        subquestion["answer"] = answer
    
    
    def new_summarization(self, model:LLMChat, savefile:str) -> None:
        for entry in tqdm(self.data, desc="Summarizing subquestions"):
            question = entry["question"]
            caption = entry["caption"]
            entities = entry["objects"]
            subquest_ans = [
                f"Subquestion {i+1}: {subquestion['question']},\nanswer: {subquestion['answer']}"
                for i, subquestion in enumerate(entry["subquestions"])
            ]
            subquest_ans_info = "\n".join(subquest_ans)
            summarized_info = summarization(entities, subquest_ans_info, question, caption, model)
            entry["summarization"] = summarized_info
            self.save_json(savefile)
            
    
    
    def subquestion_summarization(self, model:LLMChat) -> None:
        for entry in tqdm(self.data, desc="Generating subquestions"):
            summarized_content = ""
            for name, entities in entry["objects"].items():
                if len(entities) != 0:
                    for sample in entities:
                        summarized_content += summarization(sample["renamed_obj"], sample["subquestions"], model)
                        summarized_content += "\n"
            entry["summarized"] = summarized_content

    def subquestion_summarization_overall(self, model:LLMChat, savefile:str) -> None:
        for entry in tqdm(self.data, desc="Generating summarization"):
            summarized_content = summarization_all(entry["subquestions_all"], entry["question"], entry["caption"], model)

            entry["summarization"] = summarized_content
        self.save_json(savefile)

    def question_answering(self, model:LLMChat, savefile:str) -> None:
        for entry in tqdm(self.data, desc="Answering questions"):
            answer = question_answering([entry["image_path"]], entry["question"], entry["summarization"], model)
            entry["mllm_answer"] = answer
            self.save_json(savefile)
    
    
    def debate(self, debator, savefile:str) -> None:
        for entry in tqdm(self.data, desc="Debate"):
            
            llm_critic, mllm_critic, answer_after_debating = debator.run(
                image_path = entry["image_path"],
                question = entry["question"],
                caption = entry["caption"],
                subquestions = entry["subquestions_all"][0]["question"],
                subquestions_answer = entry["subquestions_all"][0]["answer"],
                summarization = entry["summarization"],
                final_answer = entry["mllm_answer"],
            )
            
            entry["llm_critic"] = llm_critic
            entry["mllm_critic"] = mllm_critic
            entry["answer_after_debating"] = answer_after_debating
            self.save_json(savefile)

    def vqa(self, model:LLMChat, savefile:str) -> None:
        for entry in tqdm(self.data, desc="Answering questions"):
            answer = vqa([entry["image_path"]], entry["question"], model)
            entry["mllm_answer"] = answer
            self.save_json(savefile)
    
    def get_dataset(self):
        return self.data
    
    
    def save_json(self, filename="save"):
        os.makedirs("save", exist_ok=True)
        with open(f"save/{filename}.json", 'w') as output_file:
            json.dump(self.data, output_file, indent=4)
        # print(f"Updated JSON file saved as {filename}")
