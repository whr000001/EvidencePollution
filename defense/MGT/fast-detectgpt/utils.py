import torch
import os
import json
import glob
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


class ProbEstimator:
    def __init__(self, ref_path='local_infer_ref'):
        self.real_crits = []
        self.fake_crits = []
        for result_file in glob.glob(os.path.join(ref_path, '*.json')):
            with open(result_file, 'r') as fin:
                res = json.load(fin)
                self.real_crits.extend(res['predictions']['real'])
                self.fake_crits.extend(res['predictions']['samples'])
        # print(f'ProbEstimator: total {len(self.real_crits) * 2} samples.')

    def crit_to_prob(self, crit):
        offset = np.sort(np.abs(np.array(self.real_crits + self.fake_crits) - crit))[100]
        cnt_real = np.sum((np.array(self.real_crits) > crit - offset) & (np.array(self.real_crits) < crit + offset))
        cnt_fake = np.sum((np.array(self.fake_crits) > crit - offset) & (np.array(self.fake_crits) < crit + offset))
        return cnt_fake / (cnt_real + cnt_fake)


def get_sampling_discrepancy_analytic(logits_ref, logits_score, labels):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref = torch.softmax(logits_ref, dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
    discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt()
    discrepancy = discrepancy.mean()
    return discrepancy.item()


def check(text, tokenizer, model):
    model.eval()
    device = model.device

    criterion_fn = get_sampling_discrepancy_analytic
    prob_estimator = ProbEstimator()

    # text = 'In a hidden valley high in the mountains of Peru, there was a little creature named Pikchu. ' \
    #        'Pikchu was not like any other animal in the valley; he was a small, ' \
    #        'bright yellow creature with pointy ears, a long tail, ' \
    #        'and cheeks that sparked with electricity when he was excited. ' \
    #        'Though he was tiny, he was curious and brave, ' \
    #        'always eager to explore the world around him.'

    # text = 'With just three months to go before the 2024 election, ' \
    #        'thousands are set to gather in Chicago this week for the Democratic National Convention. ' \
    #        'It s a tradition dating back to the 1830s, ' \
    #        'when a group of Democratic delegates supporting President Andrew Jackson gathered in Baltimore ' \
    #        'to nominate him for a second term. This year will look slightly different from others, ' \
    #        'as the Democratic Party has already officially nominated Vice-President Kamala Harris in ' \
    #        'a virtual roll call after President Joe Biden dropped out of the race.'

    tokenized = tokenizer(text, truncation=True, return_tensors="pt", padding=True,
                          return_token_type_ids=False).to(device)
    labels = tokenized.input_ids[:, 1:]
    logits_score = model(**tokenized).logits[:, :-1]

    logits_ref = logits_score

    crit = criterion_fn(logits_ref, logits_score, labels)

    prob = prob_estimator.crit_to_prob(crit)

    return prob
