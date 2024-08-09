# MirrorMirror
## Folder Structure

```
├── README.md
├── LICENSE.txt
├── environment.yml
├── benchmark
│   ├── skill folders
│   ├── annotated_benchmark (219).csv
└── src
    ├── P1
    │	├── preprocess.py
    ├── P2
    │   ├── reference_icon_gen.py
    ├── P3
    │   ├── running_clip.py
    │   ├── running_kmeans.py
    │   ├── incosistency_get_result.py
    └── running-example
        └── random_skill_each_category_1
             ├── icon
             ├── txt
             └── results


```
***Note:*** This tree includes only main files. 

## Description:

Below we describe each main file in our folder below. The three phases are detailed in Section 4 (Phase 1), Section 5 (Phase 2) and Section 6 (Phase 3).

### Environment Setup

This project required miniconda3 to run. for the convenice of establish enviroment, we prepare `environment.yml` file. After install miniconda3 from https://docs.anaconda.com/miniconda/, run conda env create -f environment.yml to import the enviroment and run conda activate environment_name (in here is sd_logo) to activate the enviroment.

This project requires Miniconda3 to execute properly. To facilitate the setup of the environment, an `environment.yml` file has been prepared. Please follow the steps below to establish and activate the environment:
1. Download and install Miniconda3 from Miniconda's official documentation.
2. Import the environment using the command: 
        ```
        conda env create -f environment.yml
        ```
3. Activate the environment by running:
        ```
        conda activate sd_logo
        ```

These steps will prepare the necessary environment to run the projects included in this repository.


### Phase 1 

`preprocess.py`: Run this file to obtain the preprocessing results of descriptions and icons. 


### Phase 2
`reference_icon_gen.py`: Run this file to obtain the Gemini output for inferring elements, and then send these elements to the Stable Diffusion model for icon generation. 

Usage:
```
reference_icon_gen.py [-h] [--api_key k] [folder_number_start] [folder_number_end]
```
Optional arguments:

        -h, --help: Show this help message and exit.
        
        --api_key k: Enter the Gemini API key (you can obtain it from https://ai.google.dev).
       
        folder_number_start: Input the folder number where reference_icon_gen.py execution begins (in the running example, it is 1).
        
        folder_number_end: Input the folder number where reference_icon_gen.py execution ends (in the running example, it is 2).

### Phase 3
`running_clip.py`: Run this file to obtain the CLIP results. It takes the preprocessed descriptions and the comparison icon group, which includes the target skill icon, unrelated icons, and generated reference icons, as multi-modal inputs, and outputs the results.

`running_kmeans.py`: Run this file to obtain the k-means results. It takes the CLIP's output as input and then applies the k-means algorithm to determine the cluster results for each skill.

`incosistency_get_result`: Run this file to obtain the final consistency checking results.





