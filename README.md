# text-clustering

## Install 

```bash
pip install scikit-learn umap-learn sentence_transformers faiss-cpu plotly matplotlib datasets
```

## Usage

Run pipeline and visualize results:

```python
from src.text_clustering import ClusterClassifier
from datasets import load_dataset

SAMPLE = 100_000

texts = load_dataset("HuggingFaceFW/FW-12-12-2023-CC-2023-06", split="train").select(range(SAMPLE))["content"]

cc = ClusterClassifier(embed_device="mps")

# run the pipeline:
embs, labels, summaries = cc.fit(texts)

# show the results
cc.show()

# save 
cc.save("./cc_100k")
```

Load classifier and run inference:
```python
from src.text_clustering import ClusterClassifier

cc = ClusterClassifier(embed_device="mps")

# load state
cc.load("./cc_100k")

# visualize
cc.show()

# classify new texts with k-nearest neighbour search
cluster_labels, embeddings = cc.infer(some_texts, top_k=1)
```

You can also run the pipeline using a script with:
```bash
# run a new pipeline
python run_pipeline.py --mode run  --save_load_path './cc_100k' --n_samples 100000 --build_hf_ds
# load existing pipeline
python run_pipeline.py --mode load --save_load_path './cc_100k' --build_hf_ds
# inference mode on new texts from an input dataset
python run_pipeline.py --mode infer --save_load_path './cc_100k'  --n_samples <NB_INFERENCE_SAMPLES> --input_dataset <HF_DATA_FOR_INFERENCE>
```
The `build_hf_ds` flag builds and pushes HF datasets, for the files and clusters, that can be directly used in the FW visualization space. In `infer` mode, we push the clusters dataset by default.

## FW topic clustering experiments
Run
```bash
# 100k samples eps 0.08 min_samples 40
python run_pipeline.py --mode run  --save_load_path './fw_100k_more_clusters' --n_samples 100000 --build_hf_ds
```
The current pipeline clusters files and asks the LLM to find the topic of each cluster and give it an educational score. We plot the distribution of samples over topics and the distribution of the educational score and save the plots in the `save_load_path` folder. To filter low files you can discrad files from clusters with a low educational score. You might need to change the response prost-processing if you decide to change the prompt.
