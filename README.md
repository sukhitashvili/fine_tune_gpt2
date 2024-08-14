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


Description of the fine-tuning procedure:
- I used HuggingFace package to load pre-trained GPT2-small model and fine-tuned it on a toy dataset generated with help of ChatGPT;
- The model was fine-tuned on prompt-to-nodes pair examples;
- The whole training script is located in the `fine_tune_gpt2.ipynb` notebook file.






