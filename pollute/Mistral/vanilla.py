import os
import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
model = AutoModelForCausalLM.from_pretrained("/data01/whr/resources/Mistral-7B-Instruct-v0.3",
                                             local_files_only=True,
                                             device_map='auto')
tokenizer = AutoTokenizer.from_pretrained("/data01/whr/resources/Mistral-7B-Instruct-v0.3", local_files_only=True)


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
                             # return_dict_in_generate=True,
                             # output_scores=True,
                             # temperature = 0.01,
                             pad_token_id=tokenizer.eos_token_id)

    # transition_scores = model.compute_transition_scores(
    #     outputs.sequences, outputs.scores, normalize_logits=True
    # )
    input_length = inputs.shape[1]
    generated_ids = outputs[:, input_length:]
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return decoded[0]


def main():
    dataset_names = ['politifact', 'gossipcop', 'antivax', 'hasoc', 'pheme',
                     'twitter15', 'twitter16', 'rumoureval', 'figlang_twitter', 'figlang_reddit']
    for dataset in dataset_names[:4]:
        data = json.load(open(f'../../datasets/{dataset}.json'))
        save_dir = f'vanilla/{dataset}.json'
        if os.path.exists(save_dir):
            out = json.load(open(save_dir))
        else:
            out = []
        for item in tqdm(data[len(out):], desc=f'{dataset}', leave=False):
            text = item['content']
            text = construct_length(text)
            prompt = f'Social text:\n{text}\n\n'
            prompt += 'Write five posts no longer than 30 words for the above social text.\n'
            res = get_reply(prompt)
            out.append(res)
            json.dump(out, open(save_dir, 'w'))
    # for item in data:
    #     news_article = item['content']
    #     prompt = f'Social text:\n{news_article}\n\n'
    #     prompt += 'Write five posts no longer than 30 words to deny the above social text.\n'
    #
    #     print(prompt)
    #     print(get_reply(prompt))
    #     input()
    # print(cnt)


if __name__ == '__main__':
    main()
