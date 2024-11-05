import json
import csv
import os

# Directories for images and JSON files
image_folder = '/home/tiennv/chaos/dataset/images'  # Replace with your image folder path
json_folder = '/home/tiennv/chaos/dataset/labels'  # Replace with your JSON folder path
output_csv = '/home/tiennv/chaos/dataset/data/dataset_deepfurniture.csv'  # Path to save the CSV file

# Initialize object counter


# Create and open the CSV file for writing
with open(output_csv, 'w', newline='') as csvfile:
    fieldnames = ['image_path', 'bbox']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Loop through each JSON file in the JSON folder
    for json_file in os.listdir(json_folder):
        if json_file.endswith('.json'):
            json_path = os.path.join(json_folder, json_file)

            # Load JSON data
            with open(json_path, 'r') as file:
                data = json.load(file)
            
            # Extract image ID from JSON and locate the corresponding image
            image_id = data.get("scene", {}).get("sceneTaskID", "")
            image_path = os.path.join(image_folder, f"{image_id}.jpg")  # Assuming images are in .jpg format

            # Extract bounding boxes, stop if object_count reaches max_objects
            instances = data.get("instances", [])
            for instance in instances:
                if 'boundingBox' in instance:
                    bbox = instance['boundingBox']
                    bbox_str = f"{bbox['xMin']},{bbox['yMin']},{bbox['xMax']},{bbox['yMax']}"
                    
                    # Write each bounding box as a separate row
                    writer.writerow({
                        'image_path': image_path,
                        'bbox': bbox_str
                    })
                    
print("CSV file created successfully .")
