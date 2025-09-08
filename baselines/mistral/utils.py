import os
import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3",
                                             device_map='auto')
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")


def construct_length(text, max_length=3840):
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
                             return_dict_in_generate=True,
                             output_scores=True,
                             # temperature = 0.01,
                             pad_token_id=tokenizer.eos_token_id)

    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, normalize_logits=True
    )
    input_length = inputs.shape[1]
    generated_ids = outputs.sequences[:, input_length:]
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return decoded[0], transition_scores.to('cpu').numpy().tolist()
