import argparse
from datasets import load_dataset


STYLES = {"wikihow":
"""Here is an extract from a webpage: "<INSERT_EXTRACT>".

Write a long and very detailed tutorial that could be part of WikiHow whose title is related to the extract above, within the context of "<INSERT_TOPIC>". Include in depth explanations for each step and how it helps achieve the desired outcome, inluding key tips and guidelines. 
Ensure clarity and practicality, allowing readers to easily follow and apply the instructions. Do not use images.""",

"textbook_narrative":
"""Here is an extract from a webpage: "<INSERT_EXTRACT>".

Write an extensive and detailed course unit suitable for a textbook, related to the given extract within the context of "<INSERT_TOPIC>". This unit should explore all pertinent concepts with in-depth explanations and technical detail. Focus on:

- Rigor: Ensure in-depth coverage of the concepts.
- Engagement: Use a narrative style akin to Michael Lewis, making it captivating and thought-provoking.
- Relevance: Connect the topic with current trends, real-life examples, or recent studies. Do not use images.""",

"textbook_academic":
"""Here is an extract from a webpage: "<INSERT_EXTRACT>".

Write an extensive and detailed course unit suitable for a textbook targeted at college students, related to the given extract within the context of "<INSERT_TOPIC>". This unit should explore all pertinent concepts with in-depth explanations and technical detail. Focus on:

- Rigor: Ensure in-depth coverage of the concepts.
- Engagement: Write with an academic, professional and engaging tone that captivates interest.
- Application: Incorporate specific, practical examples, such as proofs in calculus or critical dates and figures in history. Do not use images.""",

"blogpost":
"""Here is an extract from a webpage: "<INSERT_EXTRACT>".

Write an informative and insightful blog post that expands upon the extract above within the context of "<INSERT_TOPIC>". Your post should delve into the nuances of the topic, offering fresh perspectives or deeper analysis. Aim to:

- Inform: Provide valuable, well-researched information that educates the reader.
- Engage: Write in a conversational tone that connects with the audience, making complex ideas accessible.
- Illustrate: Use examples, anecdotes, or personal experiences to bring the topic to life.
"""
}

EXTRACT_SIZE = 1000


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, default="textbook")
    parser.add_argument("--generation_style", type=str, default="textbook_academic")
    parser.add_argument("--run_all_styles", action="store_true")
    return parser.parse_args()


def build_prompt(x, style="textbook_academic"):
    """Build the prompt based on the generation type"""
    # web extract and topic
    web_sample = x["examples"]
    web_sample = web_sample[:min(EXTRACT_SIZE, len(web_sample))]
    topic = x["category"]
    # requested generation style
    prompt = STYLES[style].replace("<INSERT_TOPIC>", topic).replace("<INSERT_EXTRACT>", web_sample)
    return {f"prompt_{style}": prompt}


if __name__ == "__main__":
    # load data=data_type and generate content in style=stayle
    args = get_args()

    print(f"Loading data fw_3M_as_{args.data_type}...")
    ds = load_dataset(f"HuggingFaceTB/fw_3M_as_{args.data_type}", split="train", num_proc=48)
    if args.run_all_styles:
        for style in STYLES.keys():
            print(f"📖 Building prompts with a {style}...")
            ds = ds.map(build_prompt, num_proc=48, fn_kwargs={"style": style})
    else:
        print(f"📖 Building prompts with a {args.generation_style}...")
        ds = ds.map(build_prompt, num_proc=48, fn_kwargs={"style": args.generation_style})
        print(ds)
    print(ds)
    ds.push_to_hub(f"HuggingFaceTB/fw_prompts_data_{args.data_type}_{args.generation_style}", private=True)
    print(f"✅ Data available at fw_prompts_data_{args.data_type}_{args.generation_style}!")

# python /fsx/loubna/projects/afaik/build_web_prompts.py --data_type textbook --generation_style textbook_academic
# python /fsx/loubna/projects/afaik/build_web_prompts.py --data_type wikihow --run_all_styles
