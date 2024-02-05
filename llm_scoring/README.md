# Evaluate the topics using different prompts after the clustering
If you already built the clusters and would like to use different prompts to get an LLM to rate their educational value, first build new evaluation prompts using:

```bash
# pull the clusters from clusters_repo and pushes the prompts to prompts_repo
# clusters_repo is the clusters dataset saved by text-clustering/run_pipeline.py
python ./build_scoring_prompts.py --clusters_repo "HuggingFaceTB/FW_clusters_100k_145_topics" --prompts_repo "HuggingFaceTB/fw_clusters_llm_judge_prompts"
```

Then use [llm-swarm](https://github.com/huggingface/llm-swarm) to generate the scores, clone and follow the installation instructions in the repository and run:

```bash
git clone https://github.com/huggingface/llm-swarm
cd llm-swarm
cp generate_scores_llm_swarm.py llm-swarm/examples/textbooks
```
```bash
python ./examples/textbooks/generate_scores_llm_swarm.py --prompt_column llm_judge_prompt --prompts_dataset HuggingFaceTB/fw_clusters_llm_judge_prompts
```
