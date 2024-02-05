import random
import argparse

from datasets import load_dataset

random.seed(10)

judge_prompt = """Evaluate the topics presented in the web examples using the additive 5-point scoring system described below.
The total score should reflect the collective educational value of all examples as a single entity, rather than scoring each example separately. Consider the examples as parts of a whole and assess their combined relevance and usefulness in an educational context. Points are accumulated based on the satisfaction of each criteria:

- Add 1 point if the topics are somewhat related to educational content or could be tangentially used in a learning environment, even if it includes some non-academic elements.
- Add another point if the topics have clear educational potential but may contain a mix of academic and non-academic content, such as advertisements, obituaries, clebrity gossip or events promotion.
- Award a third point if the topics are generally suitable for educational settings, providing useful knowledge or insights, despite containing some irrelevant or low-quality content.
- Grant a fourth point if the topics are highly relevant and beneficial for educational purposes, with minimal irrelevant content, and could be used effectively in a school or college curriculum.
- Bestow a fifth point for topics that are exceptionally educational, devoid of any inappropriate, sensitive, or non-academic material, and would significantly enhance learning in an academic setting.
<EXAMPLES>
End of the examples.

Now briefly describe the educational value of the topics discussed as a whole, avoid using enumeration or describing each example alone.
Conclude with the score using the format: 'Educational value rating: <total points>.'"""

li_prompt = """Below are extracts from some web samples. Please grade the topics discussed in these samples based on their educational relevance and appropriateness on a 5-point scale using the criteria below.
The total score should reflect the collective educational value of all examples as a single entity, rather than scoring each example separately. Consider the examples as parts of a whole and assess their combined relevance and usefulness in an educational context, using the following criteria:

1: The topics in the extracts are largely irrelevant to educational settings, containing inappropriate, controversial, or off-topic content. For instance, they might be predominantly promotional, contain irrelevant information, or focus on non-academic subjects like ads or obituaries.
2: The topics address some aspects relevant to education but do not substantially align with academic content. They might include minor non-academic elements or provide only a superficial overview of an educational subject such as events promotion.
3: The topics are generally relevant to academic settings and offer some educational value. However, they might not be comprehensive or may include slight non-academic elements. They are suitable for educational purposes but with limitations.
4: The topics are highly relevant and suitable for educational purposes. They provide substantial academic content, are well-aligned with educational settings, and contain minimal irrelevant elements. They are clear, focused, and useful for learning.
5: The topics are exemplary in their educational value. They are perfectly tailored for academic settings, offering deep, insightful, and comprehensive content. They are free from any non-academic content and reflects a high level of expertise in the subject matter.
<EXAMPLES>
End of the examples.

Briefly describe the educational value of the topics discussed as a whole, avoid using enumeration or describing each example alone.
Conclude with the score using the format: 'Educational value rating: <rating>.'"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters_repo", type=str, default="HuggingFaceTB/FW_clusters_100k_145_topics")
    parser.add_argument("--prompts_repo", type=str, default="HuggingFaceTB/fw_clusters_llm_judge_prompts")
    return parser.parse_args()


def get_cluster_text(x, max_size=2000):
    """Concatenate 10 random examples from the cluster x"""
    examples = x["examples"]
    random_ints = [random.randint(0, len(examples) - 1) for _ in range(10)]
    k = 0
    text = "\nExamples:\n"
    for i in random_ints:
        k += 1
        content = examples[i]
        size = min(len(content), max_size)
        text += f"- {content[:size]}\n\n"
    return {"cluster_examples_text": text} 

def get_evaluation_prompt(x, prompt=judge_prompt, target="llm_judge_prompt"):
    """Build evaluation prompt for cluster x from 10 random examples (we use the same for each prompt)"""
    prompt = prompt.replace("<EXAMPLES>", x["cluster_examples_text"].rstrip())
    return {target: prompt} 


if __name__ == "__main__":
    args = get_args()
    ds = load_dataset(args.clusters_repo, split="train", num_proc=2)
    ds = ds.map(get_cluster_text)
    ds = ds.map(get_evaluation_prompt, fn_kwargs={"prompt": judge_prompt, "target": "llm_judge_prompt"})
    ds = ds.map(get_evaluation_prompt, fn_kwargs={"prompt": li_prompt, "target": "backtranslation_prompt"})
    ds.push_to_hub(args.prompts_repo, private=True)

