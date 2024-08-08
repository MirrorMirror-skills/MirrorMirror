import re
import pandas as pd
from PIL import Image, ImageDraw
from pathlib import Path

# Regular expression patterns for emails and links
email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
link_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def preprocess_description(description):
    # Remove emails and links from the description
    description = re.sub(email_pattern, '', description)
    description = re.sub(link_pattern, '', description)
    return description

def process_dataframe(input_file, output_file):
    # Load DataFrame from CSV file
    df = pd.read_csv(input_file)

    # Apply preprocessing to the 'Description' column
    df['preprocess_des'] = df['Description'].apply(preprocess_description)

    # Create a new folder if it does not exist
    output_folder = Path(output_file).parent
    output_folder.mkdir(parents=True, exist_ok=True)

    # Save the preprocessed DataFrame to a new CSV file
    df.to_csv(output_file, index=False)

# Preprocessing for icon
def crop_circle(input_folder, output_folder):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    for file_name in Path(input_folder).glob('*'):
        if file_name.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            with Image.open(file_name) as img:
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

folder_number = 1
input_csv_file = f'../running-example/random_skill_each_category_{folder_number}/txt/random_skill_each_category_{folder_number}.csv'
output_csv_file = f'../running-example/random_skill_each_category_{folder_number}/txt/random_skill_each_category_{folder_number}_preprocessed.csv'
process_dataframe(input_csv_file, output_csv_file)
crop_circle(f'../running-example/random_skill_each_category_{folder_number}/icon/', f'../running-example/random_skill_each_category_{folder_number}/icon/preprocessed/')
