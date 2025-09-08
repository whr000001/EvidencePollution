import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'  # enter your own gpu idx
model_path = 'mistralai/Mistral-7B-Instruct-v0.3'  # enter your own model path
# if you are downloading models, please remove local_files_only=True
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)


def construct_length(text, max_length=3840):  # an easy way to construct the length of input text
    return text[:max_length]


@torch.no_grad()
def get_reply(prompt):
    messages = [
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
    inputs = inputs.to(model.device)

    outputs = model.generate(inputs,
                             max_new_tokens=1000,
                             do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    # to avoid the randomness through setting do_sample=False, if you want more diverse output texts, please modify this
    input_length = inputs.shape[1]
    generated_ids = outputs[:, input_length:]
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return decoded[0]
