import os
import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ['CUDA_VISIBLE_DEVICES'] = '6, 7'
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
    for dataset in dataset_names[4:]:
        data = json.load(open(f'../../datasets_sampled/{dataset}.json'))
        save_dir = f'reverse/{dataset}.json'
        if os.path.exists(save_dir):
            out = json.load(open(save_dir))
        else:
            out = []
        for item in tqdm(data[len(out):], desc=f'{dataset}', leave=False):
            news_article = item['content']
            news_article = construct_length(news_article, 2560)
            comments = item['comments']
            each_out = []
            for comment in comments:
                comment = construct_length(comment, 1280)
                prompt = f'Social text:\n{news_article}\n'
                prompt += f'Comment of this text:\n{comment}\n\n'
                prompt += 'Rewrite the comments of this text to reverse the stance of the comment.\n'
                each_out.append(get_reply(prompt))
            out.append(each_out)
            json.dump(out, open(save_dir, 'w'))


if __name__ == '__main__':
    main()
