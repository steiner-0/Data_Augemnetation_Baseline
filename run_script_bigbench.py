import argparse
import json
import asyncio
import os
from openai import AsyncOpenAI
import requests
from tqdm.asyncio import tqdm_asyncio
from api_keys import *
from prompt import *
from google import genai
from google.genai import types

# Create async client
async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

google_client = genai.Client(api_key=GOOGLE_API_KEY)

async def call_gemini_series_async(user_prompt, sys_prompt="You are a helpful assistant", model_name="gemini-1.5-flash"):
    """
    Async version of the Gemini API call to get a response for the given user prompt and system prompt.
    """
    # Most Google API clients don't have built-in async support, so we need to run the 
    # synchronous call in a thread pool to avoid blocking the event loop
    def sync_call():
        response = google_client.chat.completions.create(
            model=model_name,
            config=types.GenerateContentConfig(system_instruction=sys_prompt),
            contents=[user_prompt]
        )
        return response
    
    # Run the synchronous call in a thread pool executor
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, sync_call)
    return response

# Keep the original synchronous function for backwards compatibility if needed
def call_gemini_series(user_prompt, sys_prompt="You are a helpful assistant", model_name="gemini-1.5-flash"):
    """
    Call the Gemini API to get a response for the given user prompt and system prompt.
    """
    response = google_client.chat.completions.create(
        model=model_name,
        config=types.GenerateContentConfig(system_instruction=sys_prompt),
        contents=[user_prompt]
    )
    return response

async def call_gpt_series_async(user_prompt, sys_prompt="You are a helpful assistant", model_name="gpt-4o"):
    """
    Async version of the API call function
    """
    response = await async_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    return response

async def validate_answer_async(prediction_text, ground_truth, model_name="gpt-4o"):
    """
    Async version of the validation function
    """
    prompt = f"""
    Determine if these two answers are equivalent:
    
    Answer 1: {prediction_text}
    Answer 2: {ground_truth}
    
    You must output ONLY a single digit: 1 if the answers are equivalent (even if expressed differently), or 0 if they are not equivalent.
    """
    response = await call_gpt_series_async(prompt, model_name=model_name)
    response_content = response.choices[0].message.content.strip()
    return "1" in response_content

async def process_example(example, sys_prompt, model_name, validator_model_name, is_gemini=False):
    """
    Process a single example asynchronously
    """
    question = example['input']
    
    # Handle different output formats to get the target text (correct answer)
    if "target" in example:
        if isinstance(example["target"], list):
            # Some BIG-bench tasks have multiple valid answers
            # For simplicity, we'll just use the first one
            target_text = example["target"][0]
        else:
            target_text = example["target"]
    else:
        # Use target_scores format where score 1 indicates correct answer
        target_text = None
        for k, v in example.get("target_scores").items():
            if v == 1:
                target_text = k
                break
        
    if target_text is None:
        raise ValueError(f"Could not determine target text for example: {example}")
    
    # Make the first API call to get the model's response - now both are async
    if is_gemini:
        # Make async call for Gemini
        response = await call_gemini_series_async(
            sys_prompt=sys_prompt, 
            user_prompt=f"Answer the question:\n\n{question}",
            model_name=model_name
        )
        response_content = response.text.strip()
    else:
        # Make async call for GPT
        response = await call_gpt_series_async(
            sys_prompt=sys_prompt, 
            user_prompt=f"Answer the question:\n\n{question}",
            model_name=model_name
        )
        response_content = response.choices[0].message.content.strip()
    
    # Make the second API call to validate the answer
    correct = await validate_answer_async(response_content, target_text, model_name=validator_model_name)
    
    return {
        "question": question,
        "answer": target_text,
        "response": response_content,
        "correct": correct
    }

def _check_or_download_dataset(args):
    task_dir = os.path.join("bigbench", args.task_name)
    task_file = os.path.join(task_dir, "task.json")
    if os.path.exists(task_file):
        return json.load(open(task_file, "r"))
    os.makedirs(task_dir, exist_ok=True)
    # Construct URL to the raw task.json file on GitHub
    github_url = f"https://raw.githubusercontent.com/google/BIG-bench/main/bigbench/benchmark_tasks/{args.task_name}/task.json"
    
    # Download the file
    try:
        response = requests.get(github_url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Save the content to the file
        with open(task_file, 'wb') as f:
            f.write(response.content)
            
        print(f"Successfully downloaded {args.task_name} task file to {task_file}")
        return json.load(open(task_file, "r"))
    except Exception as e:
        raise RuntimeError(f"Failed to download {args.task_name} task file: {str(e)}")

def get_available_models():
    """
    Returns a list of available OpenAI models
    """
    return [
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-4o-mini-2024-07-18",
        "gpt-3.5-turbo",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ]

def get_sys_prompt_for_task_and_model(task_name, model_name):
    """
    Automatically selects the appropriate prompt based on task_name and model series.
    Returns the system prompt string and a flag indicating if it's a Gemini model.
    """
    is_gemini = model_name.startswith('gemini')
    
    # Map task names to their prompt variable names
    task_prompt_map = {
        "temporal_sequences": "temporal_sequences",
        "penguins_in_a_table": "penguins_in_a_table",
        "geometric_shapes": "geometric_shapes",
        "epistemic_reasoning": "epistemic_reasoning",
        "object_counting": "object_counting",
        "causal_judgment": "causal_judgment"
    }
    
    # Check if task is supported
    if task_name not in task_prompt_map:
        raise ValueError(f"Unsupported task: {task_name}")
    
    # Construct prompt variable name
    if is_gemini:
        prompt_var_name = f"BIG_{task_prompt_map[task_name]}_prompt_gemini"
    else:
        prompt_var_name = f"BIG_{task_prompt_map[task_name]}_prompt_gpt"
    
    # Get the prompt from global scope
    try:
        sys_prompt = globals()[prompt_var_name]
    except KeyError:
        raise ValueError(f"Prompt variable '{prompt_var_name}' not found in prompt.py")
    
    return sys_prompt, is_gemini

async def main_async():
    """
    Async main function to run the script with concurrent API calls
    """
    available_models = get_available_models()
    
    parser = argparse.ArgumentParser(description="Optimize a prompt for a task.")
    parser.add_argument("--task_name", type=str, default="temporal_sequences",
                        choices=["temporal_sequences", "penguins_in_a_table", "geometric_shapes",
                                 "epistemic_reasoning", "object_counting", "causal_judgment"],
                        help="The task to evaluate the model on.")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        choices=available_models,
                        help="The model to use for answering questions.")
    parser.add_argument("--validator_model", type=str, default=None,
                        choices=[m for m in available_models if m.startswith('gpt-')],  # Only GPT models for validation
                        help="The model to use for validating answers (defaults to gpt-4o if main model is Gemini).")
    parser.add_argument("--concurrency", type=int, default=40,
                        help="Number of concurrent API calls to make.")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Optional: Number of examples to sample from the dataset (default: use all)")
    
    os.makedirs("results", exist_ok=True)
    args = parser.parse_args()
    dataset = _check_or_download_dataset(args=args)
    task_name = args.task_name
    model_name = args.model
    
    # Set validator model default if not specified
    if args.validator_model is None:
        if model_name.startswith('gemini'):
            validator_model_name = "gpt-4o"  # Use GPT for validation if main model is Gemini
        else:
            validator_model_name = model_name
    else:
        validator_model_name = args.validator_model
    
    max_concurrency = args.concurrency

    # Sample the dataset if specified
    examples = dataset['examples']
    if args.sample_size and args.sample_size < len(examples):
        import random
        random.seed(42)  # For reproducibility
        examples = random.sample(examples, args.sample_size)
        print(f"Sampled {args.sample_size} examples from dataset")

    # Get the appropriate prompt and check if it's a Gemini model
    sys_prompt, is_gemini = get_sys_prompt_for_task_and_model(task_name, model_name)
    print(f"Using {'Gemini' if is_gemini else 'GPT'} prompt for {task_name}")

    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrency)

    async def process_with_semaphore(example):
        async with semaphore:
            return await process_example(example, sys_prompt, model_name, validator_model_name, is_gemini)

    # Process all examples concurrently with a semaphore to limit concurrency
    tasks = [process_with_semaphore(example) for example in examples]
    log_entries = await tqdm_asyncio.gather(*tasks, desc=f"Processing {task_name} with {model_name}")

    # Count correct answers
    correct_count = sum(1 for entry in log_entries if entry["correct"])

    # Create a results directory with timestamps to avoid overwriting
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f"results/{task_name}_{model_name}_{timestamp}"
    os.makedirs(result_dir, exist_ok=True)

    # Save log to JSON
    with open(f"{result_dir}/log.json", "w") as f:
        json.dump(log_entries, f, indent=2)

    # Save summary results to JSON
    result_summary = {
        "task": task_name,
        "model": model_name,
        "validator_model": validator_model_name,
        "total_examples": len(examples),
        "correct_examples": correct_count,
        "accuracy": correct_count / len(examples),
        "timestamp": timestamp,
        "prompt_type": "gemini" if is_gemini else "gpt",
    }

    with open(f"{result_dir}/result.json", "w") as f:
        json.dump(result_summary, f, indent=2)
    
    print(f"Task: {task_name}")
    print(f"Model: {model_name}")
    print(f"Validator: {validator_model_name}")
    print(f"Total examples: {len(examples)}")
    print(f"Correct examples: {correct_count}")
    print(f"Accuracy: {correct_count / len(examples):.2%}")
    print(f"Results saved to: {result_dir}")

def main():
    """
    Entry point that runs the async main function
    """
    asyncio.run(main_async())

if __name__ == "__main__":
    main()