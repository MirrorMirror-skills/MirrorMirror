# !pip3 install git+https://github.com/openai/CLIP.git
# !pip3 install torch
# !pip3 install transformers

import re
import pandas as pd
from PIL import Image, ImageDraw
import os
from pathlib import Path
import torch
import clip
from transformers import CLIPTokenizer

# Preprocessing for generated icons
def crop_circle(input_folder, output_folder):
    print(f"Creating directory {output_folder} and cropping images in {input_folder}")
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    for file_name in Path(input_folder).glob('*'):
        if file_name.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            with Image.open(file_name) as img:
                print(f"Processing image: {file_name}")
                width, height = img.size
                center = (width // 2 + 2, height // 2 - 46)
                radius = min(width, height) // 2 - 78

                mask = Image.new('L', (width, height), 0)
                draw = ImageDraw.Draw(mask)
                draw.ellipse((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius), fill=255)

                result = Image.new('RGB', (width, height), (255, 255, 255))
                result.paste(img, mask=mask)

                output_path = Path(output_folder) / file_name.with_suffix('.png').name
                result.save(output_path, format='PNG')
                print(f"Saved cropped image to {output_path}")

def get_image_files(directory):
    print(f"Listing image files in {directory}")
    supported_formats = ["jpg", "jpeg", "png"]  # Add or remove formats as needed
    files = os.listdir(directory)
    image_files = [file for file in files if file.split('.')[-1].lower() in supported_formats]
    return image_files

# Initialize CLIP model and device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model, preprocess = clip.load("ViT-B/32", device=device)

# Initialize the CLIP tokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# Function to preprocess and truncate text
def preprocess_and_truncate_text(text, max_length=77):
    tokens = tokenizer.tokenize(text)
    truncated_tokens = tokens[:max_length]
    truncated_text = tokenizer.convert_tokens_to_string(truncated_tokens)
    return truncated_text

for folder_number in range(1, 2):  # change folder here
    print(f"Processing folder number: {folder_number}")
    # process icons
    output_folder = f'../running-example/random_skill_each_category_{folder_number}/icon/preprocessed/'
    input_folder = f'../running-example/random_skill_each_category_{folder_number}/icon/generated/'
    crop_circle(input_folder, output_folder)

    image_directory = f'../running-example/random_skill_each_category_{folder_number}/icon/preprocessed/'
    image_files = get_image_files(image_directory)
    image_paths = [os.path.join(image_directory, file) for file in image_files]
    images = []
    image_names = []
    for path in image_paths:
        with Image.open(path) as img:
            images.append(img.copy())  # Copy the image if you need to use it outside the 'with' block
        image_names.append(os.path.basename(path))

    # Load text
    csv_file_path = f"../running-example/random_skill_each_category_{folder_number}/txt/random_skill_each_category_{folder_number}_preprocessed.csv"
    texts_df = pd.read_csv(csv_file_path)
    skills = texts_df['ASIN'].tolist()  # Assuming 'skill' is the column name
    texts = texts_df['Description'].tolist()
    print(f"Loaded texts for processing")

    # Initialize similarity matrix
    similarity_matrix = pd.DataFrame(columns=image_names, index=skills)

    # Process images and truncated texts, and compute similarity
    for i, image in enumerate(images):
        image_input = preprocess(image).unsqueeze(0).to(device)

        for j, text in enumerate(texts):
            truncated_text = preprocess_and_truncate_text(text)
            text_input = clip.tokenize([truncated_text], truncate=True).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image_input)
                text_features = model.encode_text(text_input)

                # Normalize and calculate cosine similarity
                image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
                cosine_similarity = torch.mm(image_features_norm, text_features_norm.T)

                similarity_matrix.iloc[j, i] = cosine_similarity.item()

    # Print the similarity matrix
    print(similarity_matrix)

    matrix = similarity_matrix
    csv_output_path = f'../running-example/random_skill_each_category_{folder_number}/results/distance_matrix_all_together.csv'
    matrix.to_csv(csv_output_path)
    print(f"Saved similarity matrix to {csv_output_path}")
    print("--------------------------------fin folder number: {folder_number}-----------------")
