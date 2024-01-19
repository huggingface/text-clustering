import argparse
import textwrap

import pandas as pd
from datasets import Dataset, load_dataset
from tqdm import tqdm

from src.text_clustering import ClusterClassifier


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=100_000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--save_load_path", type=str, default="./fw_afaik_topics_100k_guard_s"
    )
    parser.add_argument(
        "--input_dataset", type=str, default="HuggingFaceFW/FW-12-12-2023-CC-2023-06"
    )
    parser.add_argument(
        "--input_content", type=str, default="content"
    )
    parser.add_argument(
        "--mode",
        choices=["run", "load"],
        default="run",
        help="Run the pipeline from scratch or load existing model (default: run)",
    )
    parser.add_argument(
        "--build_hf_ds", action="store_true",
        help="Builds HF datasets used for space visualization and pushes them to the hub",
    )
    parser.add_argument(
        "--username", type=str, default="loubnabnl"
    )
    return parser.parse_args()


def build_data_clusters(cc):
    cluster_data = []
    for cluster_id, doc_ids in cc.label2docs.items():
        if cluster_id == -1:
            continue
        summary = cc.cluster_summaries.get(cluster_id, "No summary available")
        position = cc.cluster_centers.get(cluster_id, (None, None))
        examples = [cc.texts[doc_id] for doc_id in doc_ids]
        cluster_data.append(
            {
                "cluster_id": cluster_id,
                "summary": summary,
                "position": position,
                "examples": examples,
            }
        )
    df_clusters = pd.DataFrame(cluster_data)
    data_clusters = Dataset.from_pandas(df_clusters)
    return data_clusters


def build_hf_files_ds(cc, batch_size=100):
    N = len(cc.texts)
    X_values, Y_values, labels, content_display, content = [], [], [], [], []

    # Process in batches
    for i in tqdm(range(0, N, batch_size)):
        batch_texts = cc.texts[i : i + batch_size]
        X_values.extend(cc.projections[i : i + batch_size, 0])
        Y_values.extend(cc.projections[i : i + batch_size, 1])
        labels.extend(cc.cluster_labels[i : i + batch_size])
        content_display.extend([textwrap.fill(txt[:1024], 64) for txt in batch_texts])
        content.extend(batch_texts)

    df = pd.DataFrame(
        {
            "X": X_values,
            "Y": Y_values,
            "labels": labels,
            "content_display": content_display,
            "content": content,
        }
    )
    ds = Dataset.from_pandas(df)
    return ds


def main():
    args = get_args()
    cc = ClusterClassifier(embed_device=args.device)
    
    if args.mode == "run":
        # Run the pipeline
        texts = load_dataset(args.input_dataset, split="train").select(range(args.n_samples))[
            args.input_content
        ]
        _, _, summaries = cc.fit(texts)
        print(f"10 example Summaries:\n{[e for e in summaries.values()][:10]}")

        cc.save(args.save_load_path)
        print(f"Saved clusters in {args.save_load_path}.")
    else:
        # Load the pipeline
        cc.load(args.save_load_path)

    if args.build_hf_ds:
        ### Build and push HF datasets to the hub (used for the space viz)
        print("Building HF clustering datasets...")
        ds = build_hf_files_ds(cc, batch_size=100)
        print(f"files dataset {ds}")

        data_clusters = build_data_clusters(cc)
        print(f"clusters dataset {data_clusters}")

        print("Pushing to the hub")
        repo_name = args.save_load_path.split("/")[-1]
        ds.push_to_hub(f"{args.username}/{repo_name}", private=True)
        data_clusters.push_to_hub(f"{args.username}/{repo_name}_clusters", private=True)

    print("Done 🎉!")

if __name__ == "__main__":
    main()