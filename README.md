# Safe-Embed: Unveiling the Safety-Critical Knowledge of Sentence Encoders

Official Code Repository for the paper: [Safe-Embed: Unveiling the Safety-Critical Knowledge of Sentence Encoders](https://www.arxiv.org/abs/2407.06851) (KnowledgeableLMs@ACL 2024).

## Abstract

<div align="center">
  <img alt="Safe-Embed Overview" src="./images/Safe-Embed.png" width="800px">
</div>


Despite the impressive capabilities of Large Language Models (LLMs) in various tasks, their vulnerability to unsafe prompts remains a critical issue. These prompts can lead LLMs to generate responses on illegal or sensitive topics, posing a significant threat to their safe and ethical use. Existing approaches attempt to address this issue using classification models, but they have several drawbacks. With the increasing complexity of unsafe prompts, similarity search-based techniques that identify specific features of unsafe prompts provide a more robust and effective solution to this evolving problem. This paper investigates the potential of sentence encoders to distinguish safe from unsafe prompts, and the ability to classify various unsafe prompts according to a safety taxonomy. We introduce new pairwise datasets and the Categorical Purity (CP) metric to measure this capability. Our findings reveal both the effectiveness and limitations of existing sentence encoders, proposing directions to improve sentence encoders to operate as more robust safety detectors.
