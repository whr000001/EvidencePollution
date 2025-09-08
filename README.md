# EvidencePollution
This is the official repository for the paper at ACL 2025, oral: [On the Risk of Evidence Pollution for Malicious Social Text Detection in the Era of LLMs](https://arxiv.org/abs/2410.12600).

We have also uploaded the codes and related resources at the [Google Drive](https://drive.google.com/drive/folders/1B6sL_ZBz9i5RYvRqGNy0VRfrxjbgDJea?usp=sharing). If you want to directly replicate our results, we recommend downloading it directly.

## Baselines and Datasets

We have evaluate three types of existing evidence-enhanced malicious social text detectors:
- Encoder-based LMs 
  - BERT
  - DeBERTa
- LLM-based Detectors
  - Mistral
  - ChatGPT 3.5
- Existing Detectors
  - DeFEND: [[paper]](http://dl.acm.org/doi/pdf/10.1145/3292500.3330935)
  - Hyphen: [[paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/3d57795f0e263aa69577f1bbceade46b-Paper-Conference.pdf)
  - GET: [[paper]](https://arxiv.org/pdf/2201.06885)

Note that, for existing detectors, we have modified some minor details and hyperparameters to make them adaptive for our employed datasets.

If you want to employ the original versions, please refer to their own papers.

We have also uploaded our employed sampled datasets. (10 datasets, refer our paper or codes for details)

If you employ our version, please describe it correctly and cite our paper and the original paper.

## Attack
We design 13 attack methods, and you can directly download our generated polluted datasets.

Or you can generate them by running:

```
cd attack
python main.py
```

To evaluate each baseline:

For encoder-based LMs and existing detectors:
```
cd {baseline}
python train.py --dataset {dataset_name}
python inference.py --dataset {dataset_name} --pollution {pollution_name}
```
For GET, you should first run:
```
python process_data.py --dataset {dataset_name} --pollution {pollution_name}
```
For Hyphen, you should first run:
```
python amr.py --dataset {dataset_name} --pollution {pollution_name}
```
For LLM-based detectors:
```
cd {baseline}
python main.py --dataset {dataset_name} --pollution {pollution_name}
python evalaute.py --dataset {dataset_name} --pollution {pollution_name}
```

Note that, you may need to create some directories, such as checkpoints/res/amr_data, before running.

## Defense
We design three kinds of defense strategies:
### Machine-Generated Text Detection
We employ three models to identify MGTs:
- DeBERTa
- Fast-DetectGPT: [[paper]](https://openreview.net/forum?id=Bpcgcr8E8Z)
- Binocular: [[paper]](https://arxiv.org/pdf/2401.12070)

If you want to employ the original versions, please refer to their own papers.

If you employ our version, please describe it correctly and cite our paper and the original paper.

### Mixture-of-Expert (MoE)
To evaluate this strategy, pls run:
```
  python inference_moe.py --dataset {dataset_name} --pollution {pollution_name}
```

### Parameter Updating
To evaluate this strategy, pls run:
```
  python inference_updating.py --dataset {dataset_name} --pollution {pollution_name}
```


## Citation
If you find our work interesting/helpful, please consider citing this paper
```
@inproceedings{wan-etal-2025-risk,
    title = "On the Risk of Evidence Pollution for Malicious Social Text Detection in the Era of {LLM}s",
    author = "Wan, Herun  and
      Luo, Minnan  and
      Su, Zhixiong  and
      Dai, Guang  and
      Zhao, Xiang",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.480/",
    doi = "10.18653/v1/2025.acl-long.480",
    pages = "9731--9761",
    ISBN = "979-8-89176-251-0",
}
```

```
@article{wan2024risk,
  title={On the Risk of Evidence Pollution for Malicious Social Text Detection in the Era of LLMs},
  author={Wan, Herun and Luo, Minnan and Su, Zhixiong and Dai, Guang and Zhao, Xiang},
  journal={arXiv preprint arXiv:2410.12600},
  year={2024}
}
```

## Question?
Feel free to open issues in this repository! Instead of emails, GitHub issues are much better at facilitating a conversation between you and our team to address your needs. You can also contact Herun Wan through `wanherun at stu.xjtu.edu.cn`.

## Updating
### 20250908
- We have uploaded the complete resources of EvidencePollution, including codes and related data.
- We have provided a brief guideline to employ EvidencePollution.
### 20250801
- EvidencePollution is presented orally at ACL 2025!🎉🎉🎉

### 20250520
- Our paper has been accepted to the ACL 2025!🥳🥳🥳
- We plan to refine this repository.

### Before
- We have uploaded related codes. However, it's missing a lot of details.
