import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import os
import sys
from huggingface_hub import login

sys.path.append('/fred/oz479/ICT30017-P39-RAG-Watermarking-for-ownership-user-atribution/ragwm')

from src.utils import load_json
from src.models.Model import Model


class Llama(Model):
    def __init__(self, config):
        super().__init__(config)
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.device = config["params"]["device"]

        api_pos = int(config["api_key_info"]["api_key_use"])
        hf_token = config["api_key_info"]["api_keys"][api_pos]

        # Hugging Face login
        # login(token=hf_token)

        # Load tokenizer and model from the Hugging Face Hub

        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        self.model = AutoModelForCausalLM.from_pretrained(self.name, torch_dtype=torch.float16).to(self.device)

        # # If multiple GPUs are available, use DataParallel for multi-GPU support
        # print(f'torch.cuda.device_count(): {torch.cuda.device_count()}')
        # if torch.cuda.device_count() > 1:
        #     self.model = torch.nn.DataParallel(self.model, device_ids=config["params"]["device_ids"])

        # # Move model to GPU
        # self.model = self.model.to(self.device)

    def set_user_key(self, user_key):
        """user_key = bytes (the user's green-list key) or None to switch the watermark off."""
        self.user_key = user_key
        self._gl_proc = None

    def query(self, msg):
        formatted_prompt = f"[INST] {msg} [/INST]"
        input_ids = self.tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(self.device)
        gen_kwargs = {}
        # NEW: optional per-user green-word watermark (off unless set_user_key was called with a key)
        if getattr(self, "user_key", None) is not None:
            from user_greenlist import UserGreenListProcessor
            from transformers import LogitsProcessorList
            if getattr(self, "_gl_proc", None) is None:
                vocab = self.model.get_output_embeddings().weight.shape[0]
                self._gl_proc = UserGreenListProcessor(self.user_key, vocab)
            gen_kwargs["logits_processor"] = LogitsProcessorList([self._gl_proc])
        with torch.no_grad():
            outputs = self.model.generate(input_ids, temperature=self.temperature,
                                          max_new_tokens=min(self.max_output_tokens, 512),
                                          early_stopping=True, **gen_kwargs)
        out = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = out[len(formatted_prompt):]
        del input_ids, outputs
        torch.cuda.empty_cache()
        return result

    def calculate_perplexity(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"]

        if torch.cuda.is_available():
            input_ids = input_ids.to('cuda')

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        perplexity = torch.exp(loss.mean()).item()

        return perplexity


def parse_args():
    parser = argparse.ArgumentParser(description='Test Llama model query')
    parser.add_argument('--model_config_path', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='llama7b')
    parser.add_argument('--gpu_id', type=int, default=1)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)
    model_config = '/data/sunmengjie/lpz/ragwm/model_configs/llama7b_config.json'
    config = load_json(model_config)
    llm = Llama(config)
    query_prompt = "test prompt"
    response = llm.query(query_prompt)
    print(response)

    text = "This is a test sentence to calculate perplexity."
    ppl = llm.calculate_perplexity(text)
    print(f"Perplexity: {ppl}")
