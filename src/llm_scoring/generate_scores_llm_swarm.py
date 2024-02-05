import asyncio
import os
import time
from dataclasses import dataclass

import pandas as pd
from datasets import Dataset, load_dataset
from huggingface_hub import AsyncInferenceClient
from llm_swarm import LLMSwarm, LLMSwarmConfig
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoTokenizer, HfArgumentParser

HF_TOKEN = os.environ.get("HF_TOKEN", None)


@dataclass
class Args:
    max_samples: int = -1
    """The maximum umber of samples to generate (use -1 for all))"""
    tgi_instances: int = 1
    """Number of TGI instances to use"""
    model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    """Dataset containing the prompts"""
    prompts_dataset: str = "HuggingFaceTB/fw_clusters_llm_judge_prompts"
    """Model to prompt"""
    max_new_tokens: int = 1200
    """Max new tokens"""
    temperature: float = 0.6
    """Generation temperature"""
    top_p: float = 0.95
    """Generation top_p"""
    top_k: int = 50
    """Generation top_k"""
    repetition_penalty: float = 1.2
    """Generation repetition_penalty"""
    prompt_column: str = "llm_judge_prompt" 
    """Name of the column containing the prompt, choose from ["llm_judge_prompt", "backtranslation_prompt", "clustering_prompt"]"""
    repo_id: str = "HuggingFaceTB/llm_judge_responses_v3"
    """The repo id to push to"""
    push_to_hub: bool = True
    """Whether to push to hub"""


parser = HfArgumentParser((Args, LLMSwarmConfig))
args, isc = parser.parse_args_into_dataclasses()
print(args)

# overwrite model and number of instances
isc.model = args.model_name
isc.instances = args.tgi_instances
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

ds = load_dataset(
    args.prompts_dataset, token=HF_TOKEN, split="train"
).shuffle(seed=42)

if args.max_samples > 0:
    ds = ds.select(range(args.max_samples))


def postprocess_score(response):
    """Post-process model generation to get the score"""
    if "educational value rating:" in response.lower():
        score = response.lower().split("educational value rating:")[1].rstrip(".").lstrip().split(" ")[0].split("\n")[0]
    elif "educational value rating is" in response.lower():
        score = response.lower().split("educational value rating is")[1].rstrip(".").strip(":").lstrip().split(" ")[0].split("\n")[0]
    elif "educational value rating for these examples is" in response.lower():
        score = response.lower().split("educational value rating for these examples is")[1].rstrip(".").strip(":").lstrip().split(" ")[0].split("\n")[0]
    elif "educational value rating of" in response.lower():
        score = response.lower().split("educational value rating of")[1].rstrip(".").strip(":").lstrip().split(" ")[0].split("\n")[0]
    else:
        print("no score found")
        score = ""
    return score


with LLMSwarm(isc) as llm_swarm:
    semaphore = asyncio.Semaphore(llm_swarm.suggested_max_parallel_requests)
    client = AsyncInferenceClient(model=llm_swarm.endpoint)
    STOP_SEQ = ["<|endoftext|>"]

    MAX_RETRIES = 6  # maximum number of retries
    RETRY_DELAY = 4  # delay in seconds between retries

    async def process_text(sample):
        token_length = 0
        attempt = 0
        while attempt < MAX_RETRIES:
            try:
                async with semaphore:
                    completion = await client.text_generation(
                        prompt=tokenizer.apply_chat_template(
                            [{"role": "user", "content": sample[args.prompt_column]}],
                            tokenize=False,
                        ),
                        max_new_tokens=args.max_new_tokens,
                        stop_sequences=STOP_SEQ,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        repetition_penalty=args.repetition_penalty,
                    )
                    for stop_seq in STOP_SEQ:
                        if completion.endswith(stop_seq):
                            completion = completion[: -len(stop_seq)].rstrip()
                    token_length += len(tokenizer.encode(completion))
                    sample[f"{args.prompt_column}_completion"] = completion
                    sample["token_length"] = token_length
                    sample[f"{args.prompt_column}_score"] = postprocess_score(completion)
                    return sample

            except Exception as e: 
                attempt += 1
                if attempt < MAX_RETRIES:
                    print(
                        f"Request failed, retrying in {RETRY_DELAY} seconds... (Attempt {attempt}/{MAX_RETRIES})"
                    )
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    print(
                        f"Max retries reached. Failed to process the request with error {str(e)}."
                    )
                    sample[f"{args.prompt_column}_completion"] = ""
                    sample["token_length"] = 0
                    sample[f"{args.prompt_column}_score"] = ""
                    return sample

    async def main():
        start_time = time.time()
        results = await tqdm_asyncio.gather(*(process_text(sample) for sample in ds))
        end_time = time.time()
        df = pd.DataFrame(results)
        output_ds = Dataset.from_pandas(df)
        total_duration = end_time - start_time
        total_tokens = sum(output_ds["token_length"])
        overall_tokens_per_second = (
            total_tokens / total_duration if total_duration > 0 else 0
        )
        print(f"Overall Tokens per Second: {overall_tokens_per_second}")
        print(f"Generated {total_tokens / 1e6:.2f}M tokens")
        print(f"Total duration: {total_duration // 3600}h{(total_duration % 3600) // 60}min ")

        # remove empty completions
        final_data = output_ds.filter(lambda x: x[f"{args.prompt_column}_completion"] != "")
        print(output_ds)
        if args.push_to_hub:
            final_data.push_to_hub(args.repo_id, private=True)
            #failed.push_to_hub("loubnabnl/failed", private=True)

    asyncio.run(main())
