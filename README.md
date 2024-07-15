# Safe-Embed: Unveiling the Safety-Critical Knowledge of Sentence Encoders

Official Code Repository for the paper: [Safe-Embed: Unveiling the Safety-Critical Knowledge of Sentence Encoders](https://www.arxiv.org/abs/2407.06851) (KnowledgeableLMs@ACL 2024).

## Abstract

> Despite the impressive capabilities of Large Language Models (LLMs) in various tasks, their vulnerability to unsafe prompts remains a critical issue.
> These prompts can lead LLMs to generate responses on illegal or sensitive topics, posing a significant threat to their safe and ethical use. Existing approaches attempt to address this issue using classification models, but they have several drawbacks.
> With the increasing complexity of unsafe prompts, similarity search-based techniques that identify specific features of unsafe prompts provide a more robust and effective solution to this evolving problem.
> This paper investigates the potential of sentence encoders to distinguish safe from unsafe prompts, and the ability to classify various unsafe prompts according to a safety taxonomy.
> We introduce new pairwise datasets and the Categorical Purity (CP) metric to measure this capability.
> Our findings reveal both the effectiveness and limitations of existing sentence encoders, proposing directions to improve sentence encoders to operate as more robust safety detectors.

<div align="center">
  <img alt="Safe-Embed Overview" src="./images/Safe-Embed.png" width="400px">
</div>

## Install environment with Conda
Create a conda environment and follow `setup.sh` to install dependencies.

## OPENAI API KEY & Huggingface token
Put OpenAI key and Huggingface token in the `.env` file. Please avoid uploading your key and token to public repository!

## Prepare Datasets
- To create safety-contrast set, run

```
source scripts/get_contrast_set.sh
```

- You can either download safety-contrast set from [here](https://drive.google.com/drive/folders/1Yfc-hGyEkOJ5pyozrXEygBm-v2vQe0DM?usp=share_link) and put the dataset in `Safety_Contrast/data`.

## Run Experiments

### Safety-Challenging Task

- To run the task with Safety-Challenging set follow the command:

```
source scripts/safety_challenging.sh
```

- To run the task with Safety-Contrast set follow the command:

```
source scripts/safety_contrast.sh
```

### Safety-Taxonomy Task

- To run the task with Safety-Taxonomy set follow the command:

```
source scripts/safety_taxnonmy.sh
```

## Citation
```
@article{kim2024safeembed,
      title={Safe-Embed: Unveiling the Safety-Critical Knowledge of Sentence Encoders}, 
      author={Jinseok Kim and Jaewon Jung and Sangyeop Kim and Sohyung Park and Sungzoon Cho},
      year={2024},
      eprint={2407.06851},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.06851}, 
}
```