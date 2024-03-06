# Text Clustering

The Text Clustering repository contains tools to easily embed and cluster texts as well as label clusters semantically. This is repository is work in progress and serves as a minimal codebase that can be modified and adapted to other use-cases.

<center><img src="https://cdn-uploads.huggingface.co/production/uploads/61c141342aac764ce1654e43/jMKGaE_UnEfH3j8iZYXVN.png"></center>
<center>Clustering of texts in the <a href="https://huggingface.co/datasets/HuggingFaceTB/cosmopedia">Cosmopedia dataset</a>.</center>


## How it works
The pipeline consists of several distinct blocks that can be customized and the whole pipeline can run in a few minutes on a consumer laptop. Each block uses existing standard methods and work quite robustly.

<center><img src="https://huggingface.co/datasets/lvwerra/admin/resolve/main/text-clustering.png"></center>
<center>Text clustering pipeline.</center>


## Install 
Install the following libraries to get started:
```bash
pip install scikit-learn umap-learn sentence_transformers faiss-cpu plotly matplotlib datasets
```
Clone this repository and navigate to folder:
```bash
git clone https://github.com/huggingface/text-clustering.git
cd text-clustering
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

You can also change how the clusters are labeled (multiple topic (default) vs single topic with an educational score) using the flag `--topic_mode`.

## Examples

Check the `examples` folder for an example on clustering and topic labeling applied to the [AutoMathText](https://huggingface.co/datasets/math-ai/AutoMathText/) dataset, utilizing [Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia)'s web labeling approach.