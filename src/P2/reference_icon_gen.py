import textwrap
import PIL.Image
import urllib
import base64
from datetime import datetime
import pandas as pd
import os
import glob
import re
from itertools import combinations
import argparse
from pathlib import Path
import sys
import time
import emoji




def save_progress(asin, progress_file_path):
    """Append the processed ASIN to the progress file."""
    with open(progress_file_path, 'a') as f:
        f.write(f"{asin}\n")

def load_progress(progress_file_path):
    """Load processed ASINs from the progress file."""
    try:
        with open(progress_file_path, 'r') as f:
            processed_entries = f.read().splitlines()
    except FileNotFoundError:
        processed_entries = []
    return set(processed_entries)  # Returns a set of ASINs


import google.generativeai as genai
#from IPython.display import Markdown
from google.generativeai.types import HarmCategory, HarmBlockThreshold

def encode_image(img_path):
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')
    
def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

class GeminiSpeechModel:
    def __init__(self, api_key, verbose=False):
        self.api_key = api_key
        self.verbose = verbose
        self.gemini_apikey = api_key
        genai.configure(api_key = self.gemini_apikey)
        self.model_name = "gemini-1.5-pro-latest"
        self.model = genai.GenerativeModel(self.model_name)
        print("Gemini Model used: {}".format(self.model_name))
        print("Gemini apikey: ****{}".format(self.gemini_apikey[-4:]))

    def list_models(self):
        print('----------------- Available models -----------------')
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(m.name)
        print('----------------------------------------------------')

    def query(self, text, markdown=False, count_tokens=True):
        response = self.model.generate_content(
            text,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DEROGATORY: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_MEDICAL: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUAL: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_TOXICITY: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_VIOLENCE: HarmBlockThreshold.BLOCK_NONE,
            }, stream=True)
        try:
            response = response.text
        except Exception as e:
            print(f'{type(e).__name__}: {e}')
            response = ""
        
        if count_tokens:
            num_tokens = self.model.count_tokens(" ".join([text, response])).total_tokens
            if self.verbose:
                print("Token usage:\n{}".format(num_tokens))

        if self.verbose:
            print("Reply from Gemini:\n")
            if markdown:
                to_markdown(response)
            else:
                print(response)
            print("===== DONE =====")

        return (response, num_tokens)

    
# can change prompt pattern here    
def run_gemini(folder_number):
    progress_file_path = f'../running-example/random_skill_each_category_{folder_number}/txt/progress_log.txt'
    processed_entries = load_progress(progress_file_path)

    csv_path = f'../running-example/random_skill_each_category_{folder_number}/txt/random_skill_each_category_{folder_number}_preprocessed.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        print(f"File not found: {csv_path}")
        exit()

    for index, row in df.iterrows():
        asin = row['ASIN']
        if asin in processed_entries:
            print(f"Skipping {asin} as it has already been processed.")
            continue

        logfile_path = f"../running-example/random_skill_each_category_{folder_number}/txt/{asin}_gemini_log.txt"
        f = open(logfile_path, "w+")

        model = GeminiSpeechModel(args.api_key)
        #model.list_models()
        text1 = "Now, you should act like a professional app icon designer. You should accurately understand the context of an app based on its description and provide your icon generation guidance in the form of keywords. I will first send you the app descriptions, then ask a list of questions to help you understand the app context. After answering those questions, I will ask you to generate the five most representative keywords that can be used for app icon generation."
        

        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]
        
        model = genai.GenerativeModel('gemini-pro', safety_settings)
       
        
        # Max try: 10
        for attempt_no in range(10):
            
            try:

                chat = model.start_chat(history=[])
                response = chat.send_message(text1)
                time.sleep(5)

                text3 = 'Here is the app\'s long description:' + row['preprocess_des']
                response = chat.send_message(text3)
                # print(response.text)
                time.sleep(2)


                text5 = 'After reading this, you should start to answer my questions below. You should answer my questions one by one, providing answers for each question: 1. Is this app context related to specific persons? 2. Is this app designed for a specific occasion? 3. Is this app designed for a specific location? 4. What objects does this app include? 5.What is the primary functionality of this app?'
                response = chat.send_message(text5)
                # print(response.text)
                time.sleep(2)

                text8 = 'Now, based on your answers, pay close attention to the positive responses. Then, give me five ideas for generating the app icon. The icon should accurately reflect the app\'s topic and be specifically designed for this app.'
                response = chat.send_message(text8)
                # print(response.text)
                time.sleep(2)

                text9 = "List the five most important objects (persons, items, elements, activities, animals) that could appear in the app icon. Give the answer strictly following this format without any other words: the answer is [object 1, object 2, object 3, object 4, object 5]. (you need to double check that you have included the '[' and ']' in your final answer)"
                response = chat.send_message(text9)
                print(response.text)

                f.write(str(chat.history))

                print(asin+" skill finished; " + "folder number: "+str(folder_number))


                break

            except KeyboardInterrupt:
                sys.exit()
                pass
            except Exception as error:
                if attempt_no == 9:
                    sys.exit()
                print("An exception occurred when processing " + asin +\
                       ".\n" + str(error) + ' Retry ' + str(9-attempt_no) + ' times.')
                attempt_no -= 1
        f.close()

        save_progress(asin, progress_file_path)


# Function to extract contents inside the last square brackets of a text
def extract_last_bracket_content(text):
    # Find all occurrences of text enclosed in square brackets
    matches = re.findall(r'\[(.*?)\]', text)
    
    # Reverse the list to start checking from the last match
    for match in reversed(matches):
        # Check if there are exactly four commas in the match
        if match.count(',') == 4:
            return match.split(', ')
    
    # Return an empty list if no suitable match is found
    return []


# combination of icon elements (up to 2 elements)
def get_combinations(elements):
    sizes = [1, 2]  # Sizes you want the combinations for
    combination_list = []  
    for size in sizes:
        for combo in combinations(elements, size):
            combination_list.append(list(combo))  # Append the combination as a list
    return combination_list

# run stable diffision XL and get 15 icons for each skills
from diffusers import AutoPipelineForText2Image
import torch

def generate_ai_icons(combination_list, folder_number, skill_ASIN):
    pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")
    pipeline_text2image.sampler = "k_euler_ancestral"

    output_directory = f"../running-example/random_skill_each_category_{folder_number}/txt/generated/"
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    prompt_neg = "Complex, cluttered, overly detailed, intricate, ornate, busy, colorful, textured, 3D, gradient, low-resolution, pixelated, jagged, blurry, messy, inconsistent, non-uniform, chaotic, noisy, uneven, abstract, hand-drawn, sketchy, glossy, shiny, photorealistic, saturated, colorful, decorative, vintage, retro, artistic, ornamental, textured, noisy, dull, low contrast, underexposed, overexposed, poor quality, amateurish."
    
    for objects in combination_list:
        objects_string = ', '.join(objects)
        objects_string = objects_string.replace('/', ", ")

        prompt_pos = "Design a simple high-quality modern line icon of "+ objects_string +" that focus on simplicity and are suitable for app design or illustrating concepts. This icon should be created in a clean and sharp environment using a modern, minimalistic style. Utilize digital vector graphics as the medium, make them ideal for various design projects"
        print(prompt_pos)

        image = pipeline_text2image(prompt=prompt_pos, negative_prompt = prompt_neg, num_inference_steps=25).images[0]
        image_name = skill_ASIN + objects_string + ".png"
        save_path = os.path.join(output_directory, image_name)
        image.save(save_path)
        print("generated icon: " + save_path)


# deal with problematic gemini file - delete and rerun
def remove_identifier_from_progress_log(directory, identifier): 
    progress_file_path = os.path.join(directory, 'progress_log.txt')
    if os.path.exists(progress_file_path):
        with open(progress_file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        with open(progress_file_path, 'w', encoding='utf-8') as file:
            for line in lines:
                if identifier not in line:
                    file.write(line)

def contains_emoji(text):
    return emoji.emoji_count(text) > 0

def save_file_path(file_path):
    with open('emoji_file_paths.txt', 'a') as file:
        file.write(file_path + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description= 'Process folders for generating gemini logs and icons.')
    parser.add_argument('folder_range_start', type=int, help='Start of the folder range to process')
    parser.add_argument('folder_range_end', type=int, help='End of the folder range to process')
    parser.add_argument('--api_key', type=str, required=True, help='API Key for Gemini Speech Model')

    # Parse the arguments
    args = parser.parse_args()


    for folder_number in range(args.folder_range_start, args.folder_range_end + 1):
        #generate gemini logs for all skills in this folder, this step with processed-log
        run_gemini(folder_number) 
        base_directory = f"../running-example/random_skill_each_category_{folder_number}/txt/"

        # Pattern to match all files ending with '_gemini_log.txt' recursively
        file_pattern = f"{base_directory}/**/*_gemini_log.txt"

        # Find all matching files
        files = glob.glob(file_pattern, recursive=True)
        for file_path in files:
            # print(f"Processing from gemini file: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                infer_objects = extract_last_bracket_content(content)
                infer_object_list = [item.strip().strip('\'"').strip('*') for item in infer_objects]

                # Extract skill_ASIN from the filename
                skill_ASIN = file_path.split('/')[-1].split('_gemini_log.txt')[0]

                #extract icon objects from gemini answers and put into list
                print("Extracted list from the last answer:", infer_object_list) 
                  
                #create objects combination list
                combination_list = get_combinations(infer_object_list)
                print(combination_list)
                print(len(combination_list))

                #generate icons for each combination
                generate_ai_icons(combination_list, folder_number, skill_ASIN)











