import json
import random
import os



def add_random_files(input_json_path, output_json_path, directory_path):
    """
    Load JSON, add 3 random filenames from content image directory to each entry.
    
    Args:
        input_json_path: Path to input JSON file
        output_json_path: Path to output JSON file
        directory_path: Directory to get random files from
    """
    # Get all files from the directory
    files = [f for f in os.listdir(directory_path) 
             if os.path.isfile(os.path.join(directory_path, f))]
    
    # Load JSON data
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    # Add 3 random files to each entry
    for entry in data:
        entry['random_images'] = [os.path.join(directory_path, f) for f in random.sample(files, 3)]
    
    # Save back to JSON
    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Processed {len(data)} entries")
    print(f"Saved to {output_json_path}")


if __name__ == "__main__":
    input_file = "audio_image_pairs.json"
    output_file = "audio_style_content_pairs.json"
    random_dir = "/graphics/scratch2/students/reutemann/train2017"
    
    add_random_files(input_file, output_file, random_dir)