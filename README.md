# GPT2 Fine-Tuning Repository

The fine-tuned GPT2 bot is hosted here (The bot might be in sleep mode, so please awake it up and wait a bit until it loads): [Link](https://superbot-gpt2.streamlit.app/)


## Commands to host the bot locally:
1. Git clone the repository;
2. You can run/host the bot via Docker or manually (CPU device will be used for processing):
   1. To host via docker use the commands: 
      1. `docker build -t my-python-app .`
      2. `docker run -d -p 4000:80 my-python-app`
   2. To host manually you need Python3.9 to have installed and use commands:
      1. `pip install -r requirements.txt`
      2. `streamlit run app.py`

## Description of the files in the repo
1. `fine_tune_gpt2.ipynb` contains all the training code and prompts at the bottom, used to prompt ChatGPT to generated data;
2. `best_val_rouge1_model.pt` Trained model GPT2 small;
3. `data.csv` data CSV file generated via ChatGPT;
4. `app.py` contains code to define UI using Streamlit package;
5. `model.py` contains text processing code by GPT2;


## Description of the fine-tuning procedure:
- I used HuggingFace package to load pre-trained GPT2-small model and fine-tuned it on a toy dataset generated with help of ChatGPT;
- The model was fine-tuned on prompt-to-nodes pair examples;
- The whole training script is located in the `fine_tune_gpt2.ipynb` notebook file.

## Possible future improvements/experiments:
- Larger models with customized smaller context size can be trained on this dataset. Because average prompt length in token space is 
around 35-40 tokens, I think usage of 1024 context length is unnecessary. Additional, with will reduce model slightly as positional encoding
can be reduced let's say down to 512 context length.
- If bigger models are used, keeping they previously leaned knowledge would be more beneficial and thus methods like LORA should be used.
- Model quantization for inference speed-up and memory usage reduction.






