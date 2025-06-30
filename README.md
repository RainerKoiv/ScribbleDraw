# Project description
Project author: Rainer Kõiv

Project supervisor: Ardi Tampuu

This project was developed as a BSc final thesis. The accompanying demo application is intended for display at the Institute of Computer Science, Delta Building, University of Tartu, Estonia.

The demo allows users to create simple sketches and enhance them using AI-powered image generation. Users can choose a generation mode (task type), a desired visual style, and a background. 
The user's drawing will be processed with a ControlNet model. A VLM (Vision-Language Model) will detect an object or generate a description based on the drawing. A new image will then be generated based on the user's sketch and the generated description.

Release 1.0.0 is the first version developed as the final thesis.
Release 2.0.0 is a simplified version and customized to be displayed at the AHHAA AI exhibit and also the Delta Center.

## Screenshots
User Interface (version 2.0.0)

English
![image](https://github.com/user-attachments/assets/48f2dc44-a652-4e14-ac26-c0f4fe69fa4c)

Estonian
![image](https://github.com/user-attachments/assets/8c37fea9-2ae9-4da3-adaa-761f5b4710fc)


Example (processed) inputs and outputs

![image](https://github.com/user-attachments/assets/bb863385-c2b0-4026-981a-dc19a27677bc)

## Demo video (version 1.0.0)
https://github.com/user-attachments/assets/c249b028-c3db-4011-9c9c-7bea1d14491d



## Installation
Requirements:
- High performance GPU, like Nvidia RTX 3060ti or 3080
- Pytorch, CUDA toolkit (12.8), cuDNN 12.x
   - https://www.youtube.com/watch?v=r7Am-ZGMef8
   - https://www.youtube.com/watch?v=c0Z_ItwzT5o
- Python 3 (3.12.7)

How to get started:
- Clone the repo
    - If you are interested in the first version, then checkout to branch with the tag 1.0.0
- Activate python virtual environment ``source .venv/bin/activate`` and go into the project folder
- First time installations:
    - pip install diffusers["torch"] transformers
    - pip install controlnet_aux
    - pip install diffusers transformers accelerate
    - pip install mediapipe
    - pip install gradio==5.15.0
- Run the front-end: ``uvicorn frontend:app --reload``, wait for it to download everything and open the local URL (http://127.0.0.1:8000) given in the terminal

## Usage
The project version 2.0.0 UI is divided into two sides:
  - Left side (inputs):
    - Sketchpad - user can draw a picture
    - Style buttons - user can select the desired style
    - Generate Image button
  - Right side (outputs):
    - Generated image with the selected style
    - Additional info about the scene description and some tips

## Models Used

This project uses the following pretrained models:

- [Florence-2-large](https://huggingface.co/microsoft/Florence-2-large) by Microsoft
  -   ```@article{xiao2023florence,
      title={Florence-2: Advancing a unified representation for a variety of vision tasks},
      author={Xiao, Bin and Wu, Haiping and Xu, Weijian and Dai, Xiyang and Hu, Houdong and Lu, Yumao and Zeng, Michael and Liu, Ce and Yuan, Lu},
      journal={arXiv preprint arXiv:2311.06242},
      year={2023}
      }
- [Stable Diffusion v1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) by Stability AI
  -     @InProceedings{Rombach_2022_CVPR,
        author    = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj\"orn},
        title     = {High-Resolution Image Synthesis With Latent Diffusion Models},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        month     = {June},
        year      = {2022},
        pages     = {10684-10695}
        }


- [ControlNet (scribble)](https://huggingface.co/lllyasviel/sd-controlnet-scribble) by lllyasviel
  - ```@misc{zhang2023adding,
    title={Adding Conditional Control to Text-to-Image Diffusion Models},
    author={Lvmin Zhang and Maneesh Agrawala},
    year={2023},
    eprint={2302.05543},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
    }
Please refer to each model's license and citation requirements on their respective Hugging Face pages.
