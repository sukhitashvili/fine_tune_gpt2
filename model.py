import pickle
from pathlib import Path
from typing import List

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
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
        self.SPECIAL_TOKEN = '>>'
        # Load the GT token ids list from the file
        with open(str(Path(__file__).parent / 'gt_token_ids.pkl'), 'rb') as file:
            self.gt_token_id_list = pickle.load(file)
        self.gt_token_ids_set = set(self.gt_token_id_list)
        self.gt_token_threshold = 0.5  # TODO: check how this value affect rouge scores
        self.default_response = ("Sorry, I cannot understand you prompt :(  \n"
                                 "Please, provide prompts only related to the node generation topic")

    def predict(self, text_input: str) -> str:
        """
        Generates model's response
        Args:
            text_input: User query

        Returns:
            Generated text
        """
        text_input += self.SPECIAL_TOKEN
        output_text = self.generate_replay(text_input=text_input)
        return output_text

    @torch.no_grad()
    def generate_replay(self, text_input: str):
        """
        Generates model's response and checks if response is strictly about topic of nodes.
        Args:
            text_input: User query

        Returns:
            Generated text
        """
        input_ids = self.tokenizer.encode(text_input, return_tensors='pt')
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
        pred_labels = decoded_output.split(self.SPECIAL_TOKEN)[1].strip()

        # Check if the output contains correct Node tokens
        if not self.contains_valid_tokens(token_ids=self.tokenizer.encode(pred_labels)):
            return self.default_response
        return pred_labels

    def contains_valid_tokens(self, token_ids: List[int]) -> bool:
        """
        Validates is generated tokens are part of nodes' token ids.
        If generated tokens are outside of this space, this means model has generated response to random query or
        it was hallucinating.
        Args:
            token_ids:

        Returns:

        """
        gen_token_set = set(token_ids)
        diff = gen_token_set - self.gt_token_ids_set
        if (len(diff) / len(gen_token_set)) < self.gt_token_threshold:
            return True
        return False


class FlanT5:
    def __init__(self, model_file_path: str):
        self.device = torch.device('cpu')
        model_name = 'google/flan-t5-small'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(original_model, model_id=model_file_path, is_trainable=False,
                                          torch_dtype=torch.bfloat16)
        model.merge_and_unload()  # merge lora layers back into weights matrix
        model.to(self.device)
        self.model = model
        self.model.eval()

    def predict(self, text_input: str) -> str:
        """
        Generates model's response
        Args:
            text_input: User query

        Returns:
            Generated text
        """
        output_text = self.generate_replay(text_input=text_input)
        return output_text

    @torch.no_grad()
    def generate_replay(self, text_input: str):
        """
        Generates model's response and checks if response is strictly about topic of nodes.
        Args:
            text_input: User query

        Returns:
            Generated text
        """
        input_ids = self.tokenizer.encode(text_input, return_tensors='pt')
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
        return decoded_output


model_path = str(Path(__file__).parent / 't5_adapter_ckpt')
t5_model = FlanT5(model_file_path=model_path)
# response = t5_model.predict("Navigate to a different URL after 5 seconds when a key is pressed")
# print(response)
