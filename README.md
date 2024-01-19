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
python run_pipeline.py --mode run --n_samples 100000 --save_load_path './cc_100k' --build_hf_ds
# build_hf_ds builds and pushes HF datasets to be used in a visualization space
# switch to --mode load to use an existing clustering instead of running a new one
python run_pipeline.py --mode load --save_load_path './cc_100k' --build_hf_ds
```