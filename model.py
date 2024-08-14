from pathlib import Path

import torch
from transformers import GPT2Tokenizer

torch.set_default_dtype(torch.bfloat16)


class GPT2:
    def __init__(self, model_file_path: str):
        self.device = torch.device('cpu')
        self.model = torch.load(model_file_path, map_location=torch.device('cpu')).to(self.device)
        self.model.eval()
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        # Added padding token like during training
        PADDING_TOKEN = "<|PAD|>"
        special_tokens_dict = {'pad_token': PADDING_TOKEN}
        self.tokenizer.add_special_tokens(special_tokens_dict)

    def predict(self, text_input: str) -> str:
        output_text = self.generate_replay(text_input=text_input)
        return output_text

    @torch.no_grad()
    def generate_replay(self, text_input: str):
        input_ids = self.tokenizer(text_input, return_tensors='pt')
        output = self.model.generate(input_ids=input_ids.to(self.device),
                                     do_sample=True,
                                     max_new_tokens=10,
                                     pad_token_id=self.tokenizer.pad_token_id,
                                     eos_token_id=self.tokenizer.eos_token_id,
                                     top_k=20,
                                     top_p=0.95,
                                     )
        # Decode the output
        decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        pred_labels = decoded_output.split('>>')[1].strip()
        return pred_labels


model_path = str(Path(__file__).parent / 'best_val_rouge1_model.pt')
gpt2_model = GPT2(model_file_path=model_path)
