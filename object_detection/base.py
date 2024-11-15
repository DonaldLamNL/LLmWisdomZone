import json
import cv2  # OpenCV for image processing
import os
from ultralytics import YOLO


class ObjectDetection:
    def __init__(self, model_path, custom_classes=None):
        self.model = YOLO(model_path)
        if custom_classes:
            self.model.set_classes(custom_classes)

    def predict(self, image_path, target_entities):
        self.model.set_classes(target_entities)
        results = self.model.predict(image_path)
        entity_boxes = {entity: [] for entity in target_entities}

        for result in results:
            for box, class_id in zip(result.boxes.xyxy, result.boxes.cls):
                entity_name = self.model.names[int(class_id)]
                if entity_name in entity_boxes:
                    entity_boxes[entity_name].append(box.tolist())

        return entity_boxes

    def crop_and_save_images(self, image_path, entity_boxes, output_paths_cropped, output_paths_box):
        image = cv2.imread(image_path)

        for entity, boxes in entity_boxes.items():
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)  # Convert to integers
                cropped_image = image[y1:y2, x1:x2]  # Crop the image
                cv2.imwrite(output_paths_cropped[entity][i], cropped_image)  # Save the cropped image

                # Draw the bounding box on the original image for visualization
                image_with_box = image.copy()  # Create a copy of the original image
                cv2.rectangle(image_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box with thickness 2
                cv2.imwrite(output_paths_box[entity][i], image_with_box)


def process_json(input_json_file, output_json_file, model_path):
    box_image_folder = "pope_data/box_image"
    org_image_folder = "pope_data/org_image"

    with open(input_json_file, 'r') as file:
        data = json.load(file)

    detector = ObjectDetection(model_path)

    for item in data:
        image_source = item["image_source"]
        image_path = os.path.join(org_image_folder , image_source + ".jpg")
        item['image_path'] = image_path
        objects = item["objects"]

        target_entities = list(objects.keys())
        entity_boxes = detector.predict(image_path, target_entities)
        output_paths_cropped_dict = {entity: [] for entity in target_entities}
        output_paths_box_dict = {entity: [] for entity in target_entities}

        # Update the objects in the JSON with generated results
        for entity in target_entities:
            count = 1  # To track the number of objects detected for each entity
            output_paths_cropped = []
            output_paths_box = []

            # Create output paths and update JSON objects
            for box in entity_boxes.get(entity, []):
                renamed_obj = f"{entity}_{count}"
                # Generate the output path
                # output_path = f"{os.path.splitext(image_path)[0]}_{renamed_obj}.jpg"
                cropped_image_path = f"{box_image_folder}/{os.path.splitext(os.path.basename(image_path))[0]}_{renamed_obj}.jpg"
                box_image_path = f"{box_image_folder}/{os.path.splitext(os.path.basename(image_path))[0]}_{renamed_obj}_with_box.jpg"
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

                if entity in item["objects"]:
                    item["objects"][entity].append(new_obj)
                else:
                    item["objects"][entity] = [new_obj]

                count += 1

            # Create the output directory if it doesn't exist
            for path in output_paths_cropped:
                os.makedirs(os.path.dirname(path), exist_ok=True)
            for path in output_paths_box:
                os.makedirs(os.path.dirname(path), exist_ok=True)
            output_paths_cropped_dict[entity] = output_paths_cropped
            output_paths_box_dict[entity] = output_paths_box

        # Crop and save images based on predicted bounding boxes
        detector.crop_and_save_images(image_path, entity_boxes=entity_boxes,
                                      output_paths_cropped=output_paths_cropped_dict,
                                      output_paths_box=output_paths_box_dict)

    # Save the updated data back to the JSON file
    with open(output_json_file, 'w') as file:
        json.dump(data, file, indent=4)


# Example usage
# input_json_file = "pope_data.json"
# output_json_file = "after_obj_detect_data.json"
# model_path = "yolov8s-world.pt"
# process_json(input_json_file, output_json_file, model_path)
