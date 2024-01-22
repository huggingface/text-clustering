import argparse
import textwrap

import pandas as pd
import numpy as np
from datasets import Dataset, load_dataset

from src.text_clustering import ClusterClassifier


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=100_000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_load_path", type=str, default="./cc_100k")
    parser.add_argument(
        "--input_dataset", type=str, default="HuggingFaceFW/FW-12-12-2023-CC-2023-06"
    )
    parser.add_argument("--input_content", type=str, default="content")
    parser.add_argument(
        "--mode",
        choices=["run", "load", "infer"],
        default="run",
        help="Run the pipeline from scratch/load existing model to build hf datasets or to infer on new texts",
    )
    parser.add_argument(
        "--inference_repo_name",
        type=str,
        default="infer_fw_on_ultrachat",
        help="HF repo name for the clusters dataset in inference mode",
    )
    parser.add_argument(
        "--build_hf_ds",
        action="store_true",
        help="Builds HF datasets used for space visualization and pushes them to the hub",
    )
    parser.add_argument("--username", type=str, default="loubnabnl")
    return parser.parse_args()


def build_hf_data_clusters(cc, texts=None, labels=None):
    """
    Build an HF dataset containing information on each cluster.

    Args:
        cc: ClusterClassifier object.
        texts: list of texts used for inference mode.
        labels: list of cluster labels corresponding to the texts for inference mode.

    If `texts` and `labels` are not provided, the function will use the data available in `cc`
    to construct the dataset. Otherwise it will run in inference mode on texts.
    """
    cluster_data = []
    for cluster_id in cc.label2docs.keys():
        if cluster_id == -1:
            continue

        # inference mode
        if texts is not None and labels is not None:
            labels_array = np.array(labels)
            files_in_cluster = np.where(labels_array == cluster_id)[0]
            examples = [texts[doc_id] for doc_id in files_in_cluster]
        else:
            doc_ids = cc.label2docs[cluster_id]
            examples = [cc.texts[doc_id] for doc_id in doc_ids]

        cluster_info = {
            "cluster_id": cluster_id,
            "summary": cc.cluster_summaries[cluster_id],
            "examples": examples,
        }

        if not texts:
            cluster_info["position"] = cc.cluster_centers[cluster_id]

        cluster_data.append(cluster_info)

    return Dataset.from_pandas(pd.DataFrame(cluster_data))


def build_hf_data_files(cc):
    """
    Build an HF dataset containing information on each file and the cluster they belong to
    """

    df = pd.DataFrame(
        data={
            "X": cc.projections[:, 0],
            "Y": cc.projections[:, 1],
            "labels": cc.cluster_labels,
            "content_display": [textwrap.fill(txt[:1024], 64) for txt in cc.texts],
        }
    )
    return Dataset.from_pandas(df)


def build_and_push(cc, args):
    """Build HF files & clusters datasts and push them to the hub"""
    print("Building HF datasets...")
    ds = build_hf_data_clusters(cc)
    data_clusters = build_hf_data_files(cc)
    print(f"Files dataset {ds}\nClusters dataset {data_clusters}")

    repo_name = args.save_load_path.split("/")[-1]
    print(f"Pushing to the hub at {repo_name}...")
    ds.push_to_hub(f"{args.username}/{repo_name}", private=True)
    data_clusters.push_to_hub(f"{args.username}/{repo_name}_clusters", private=True)


def main():
    args = get_args()
    cc = ClusterClassifier(embed_device=args.device)

    if args.mode == "run":
        # Run a new pipeline on texts
        texts = load_dataset(args.input_dataset, split="train", token=True).select(
            range(args.n_samples)
        )[args.input_content]

        _, _, summaries = cc.fit(texts)
        print(f"10 example Summaries:\n{[e for e in summaries.values()][:10]}")

        cc.save(args.save_load_path)
        print(f"Saved clusters in {args.save_load_path}.")

        if args.build_hf_ds:
            build_and_push(cc, args)

    elif args.mode == "infer":
        # Run inference mode on texts using an existing pipeline
        cc.load(args.save_load_path)
        print(
            f"Running inference on {args.n_samples} samples of {args.input_dataset} using clusters in {args.save_load_path}."
        )
        texts = load_dataset(args.input_dataset, split="train", token=True).select(
            range(args.n_samples)
        )[args.input_content]
        cluster_labels, _ = cc.infer(texts, top_k=1)

        ds = build_hf_data_clusters(cc, texts, cluster_labels)
        target_repo = {args.username} / {args.inference_repo_name}
        print(f"Pushing to hub at {target_repo}...")
        ds.push_to_hub(f"{target_repo}", private=True)

    else:
        # Load existing pipeline
        if args.build_hf_ds:
            cc.load(args.save_load_path)
            build_and_push(cc, args)
        else:
            print("Using mode=load but build_hf_ds is False, nothing to be done.")

    print("Done 🎉")


if __name__ == "__main__":
    main()
