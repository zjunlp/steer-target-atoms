
import sys
# sys.path.append("../../")
sys.path.append("../")
# sys.path.append("./")


import torch
import argparse
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from baseline.caa.utils.input_format import llama3_chat_input_format

def load_model_and_tokenizer(model_path, tokenizer_path=None):
    if tokenizer_path == None:
        tokenizer_path = model_path

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if "gemma" in model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True, padding_side="right"
        )
    if "llama" in model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True, padding_side="left"
        )
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def batch_infer(model, tokenizer, prompts, batch_size=4, max_new_tokens=20, return_score=False):
    
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    answers = []
    for batch_input in tqdm(batch_split(prompts, batch_size)):
        encode_inputs = prepare_input(tokenizer, batch_input, device=model.device)
        outputs = model.generate(
            **encode_inputs,
            return_dict_in_generate=True,
            output_scores=True,
            **generation_config,
        )
        input_length = encode_inputs.input_ids.shape[1]
        generated_tokens = outputs.sequences[:, input_length:]
        generated_texts = tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )

        if return_score:
            scores = outputs.scores[0].cpu().detach().numpy()
            
            for i in range(len(generated_texts)):
                answers.append({
                    # "pred": generated_texts[i],
                    "score": scores[i]
                })
        else:
            answers.extend(generated_texts)

    return answers


def batch_split(prompts, batch_num):
    batch_prompts = []
    mini_batch = []
    for prompt in prompts:
        mini_batch.append(prompt)
        if len(mini_batch) == batch_num:
            batch_prompts.append(mini_batch)
            mini_batch = []
    if len(mini_batch) != 0:
        batch_prompts.append(mini_batch)
    return batch_prompts


def prepare_input(tokenizer, prompts, device="cuda"):
    input_tokens = tokenizer.batch_encode_plus(
        prompts, return_tensors="pt", padding=True
    )
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(device)

    return input_tokens


def clean_preds(preds):
    final_preds = []
    split_text = [
        "\n\n\n<end_of_turn>\n<eos>", 
        "<end_of_turn>",
        "<eos>",
        "\n\n",
        "System:",
        "Question:",
        "Answer:"
    ]
    for pred in preds:
        pred = pred.strip()
        for sp in split_text:
            pred = pred.split(sp)[0]
        final_preds.append(pred.strip())

    return final_preds

def clean_preds_yn(preds):
    final_preds = []
    split_text = [
        "\n",
        ",",
        "."
    ]
    for pred in preds:
        pred = pred.strip()
        for sp in split_text:
            pred = pred.split(sp)[0]
        final_preds.append(pred.strip())

    return final_preds
