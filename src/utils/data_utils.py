import os
import random
from typing import Optional, List

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


# Only for evaluation
def get_wikitext2(tokenizer: AutoTokenizer,  sequence_length: int):
    test_dataset_raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    test_dataset_tok = tokenizer("\n\n".join(test_dataset_raw["text"]), return_tensors="pt").input_ids
    num_test_sequences = test_dataset_tok.numel() // sequence_length
    test_loader = []
    for i in range(num_test_sequences):
        test_loader.append(test_dataset_tok[:, i * sequence_length : (i + 1) * sequence_length])
    return test_loader


def get_c4(
    tokenizer: AutoTokenizer, 
    max_sequence_length: int,
    num_calibration_samples: Optional[int] = None,
    seed: int = 42
):
    train_datasetraw = load_dataset(
        'allenai/c4', 
        data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, 
        split='train'
    )
    all_id =[]
    trainloader = []
    for _ in range(num_calibration_samples):
        while True:
            i = random.randint(0, len(train_datasetraw) - 1)
            if i in all_id:
                continue
            trainenc = tokenizer(train_datasetraw[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= max_sequence_length:
                break
        all_id.append(i)
        i = random.randint(0, trainenc.input_ids.shape[1] - max_sequence_length)
        tokenized_sample = trainenc.input_ids[:, i:i + max_sequence_length]
        trainloader.append(tokenized_sample)
    return trainloader


def get_open_thoughts(
    tokenizer: AutoTokenizer, 
    max_sequence_length: int,
    num_calibration_samples: Optional[int] = None,
    seed: int = 42
) -> List[torch.Tensor]:
    train_dataset_raw = load_dataset("open-thoughts/OpenThoughts-114k", split="train")
    if num_calibration_samples:
        train_dataset_raw = train_dataset_raw.shuffle(seed=seed).select(range(num_calibration_samples))
    # Preprocess the data into the format the model is trained with.
    def preprocess(example):
        messages = []
        # add system prompt
        messages.append({"role": "system", "content": example['system']})
        # add dialogue
        for message in example['conversations']:
            messages.append({"role": message["from"], "content": message["value"]})
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}
    train_dataset_raw = train_dataset_raw.map(preprocess)
    # Tokenize the data
    def tokenize(sample):
        return tokenizer(
            sample["text"], 
            padding=False, 
            max_length=max_sequence_length, 
            truncation=True, 
            add_special_tokens=False,
        )
    train_dataset = train_dataset_raw.map(tokenize, remove_columns=train_dataset_raw.column_names)
    train_dataset = [torch.tensor(sample['input_ids']).unsqueeze(0) for sample in train_dataset]
    return train_dataset


def get_open_platypus(
    tokenizer: AutoTokenizer, 
    max_sequence_length: int,
    num_calibration_samples: Optional[int] = None,
    seed: int = 42
) -> List[torch.Tensor]:
    train_dataset_raw = load_dataset("garage-bAInd/Open-Platypus", split="train")
    if num_calibration_samples:
        train_dataset_raw = train_dataset_raw.shuffle(seed=seed).select(range(num_calibration_samples))
    # Preprocess the data into the format the model is trained with.
    def preprocess(example):
        messages = [
            {"role": "user", "content": example["instruction"]}, 
            {"role": "assistant", "content":  example["output"]},
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}
    train_dataset_raw = train_dataset_raw.map(preprocess)
    # Tokenize the data
    def tokenize(sample):
        return tokenizer(
            sample["text"], 
            padding=False, 
            max_length=max_sequence_length, 
            truncation=True, 
            add_special_tokens=False,
        )
    train_dataset = train_dataset_raw.map(tokenize, remove_columns=train_dataset_raw.column_names)
    train_dataset = [torch.tensor(sample['input_ids']).unsqueeze(0) for sample in train_dataset]
    return train_dataset

def get_fineweb_edu(
    tokenizer: AutoTokenizer, 
    max_sequence_length: int,
    num_calibration_samples: Optional[int] = None,
    seed: int = 42
) -> List[torch.Tensor]:
    train_dataset_raw = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    train_dataset_raw = train_dataset_raw.shuffle(seed=seed, buffer_size=1_000)
    trainloader = []
    for j, sample in enumerate(train_dataset_raw):
        trainenc = tokenizer(
            sample['text'],
            return_tensors="pt"
        )
        if trainenc.input_ids.shape[1] < max_sequence_length:
            continue
        i = random.randint(0, trainenc.input_ids.shape[1] - max_sequence_length)
        tokenized_sample = trainenc.input_ids[:, i:i + max_sequence_length]
        trainloader.append(tokenized_sample)
        if len(trainloader)>=num_calibration_samples:
            break
    return trainloader

def get_ultrachat_200k(
    tokenizer: AutoTokenizer, 
    max_sequence_length: int,
    num_calibration_samples: Optional[int] = None,
    seed: int = 42
) -> List[torch.Tensor]:
    train_dataset_raw = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    if num_calibration_samples:
        train_dataset_raw = train_dataset_raw.shuffle(seed=seed).select(range(num_calibration_samples))
    # Preprocess the data into the format the model is trained with.
    def preprocess(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}
    train_dataset_raw = train_dataset_raw.map(preprocess)
    # Tokenize the data
    def tokenize(sample):
        return tokenizer(
            sample["text"], 
            padding=False, 
            max_length=max_sequence_length, 
            truncation=True, 
            add_special_tokens=False,
        )
    train_dataset = train_dataset_raw.map(tokenize, remove_columns=train_dataset_raw.column_names)
    train_dataset = [torch.tensor(sample['input_ids']).unsqueeze(0) for sample in train_dataset]
    return train_dataset

def get_tulu3_sft_mixture(
    tokenizer: AutoTokenizer, 
    max_sequence_length: int,
    num_calibration_samples: Optional[int] = None,
    seed: int = 42
) -> List[torch.Tensor]:
    # Load raw dataset
    train_dataset_raw = load_dataset("allenai/tulu-3-sft-mixture", split="train")

    # Optionally subsample early for efficiency
    if num_calibration_samples:
        train_dataset_raw = train_dataset_raw.shuffle(seed=seed).select(range(num_calibration_samples))

    # Preprocess into chat text
    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"], tokenize=False
            )
        }

    train_dataset_raw = train_dataset_raw.map(preprocess)

    # Tokenize into input_ids
    def tokenize(sample):
        tokenized = tokenizer(
            sample["text"],
            padding=False,
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=False,
        )
        return tokenized

    tokenized_dataset = train_dataset_raw.map(tokenize, remove_columns=train_dataset_raw.column_names)

    # Convert to list of tensors
    train_dataset = [torch.tensor(sample['input_ids']).unsqueeze(0) for sample in tokenized_dataset]

    return train_dataset


def get_local_jsonl(
    json_path: str,
    tokenizer: AutoTokenizer,
    max_sequence_length: int,
    num_calibration_samples: Optional[int] = None,
    seed: int = 42
) -> List[torch.Tensor]:
    """
    Load a local .jsonl/.json file with {"text": "..."} per line and return
    a list of (1, max_sequence_length) tensors for calibration.
    Uses streaming to avoid loading the whole file into memory.
    """
    if num_calibration_samples is None:
        raise ValueError("num_calibration_samples must be provided for local jsonl calibration.")

    ds = load_dataset(
        "json",
        data_files=json_path,
        split="train",
        streaming=True,
    )
    # Shuffle for better diversity; buffer_size can be small because your file isn't huge
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    trainloader: List[torch.Tensor] = []
    for sample in ds:
        text = sample.get("text", "")
        if not text:
            continue
        trainenc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        if trainenc.input_ids.shape[1] < max_sequence_length:
            continue
        i = random.randint(0, trainenc.input_ids.shape[1] - max_sequence_length)
        tokenized_sample = trainenc.input_ids[:, i:i + max_sequence_length]
        trainloader.append(tokenized_sample)
        if len(trainloader) >= num_calibration_samples:
            break

    if len(trainloader) < num_calibration_samples:
        raise RuntimeError(
            f"Not enough samples from local jsonl: got {len(trainloader)} < {num_calibration_samples}. "
            f"File={json_path}"
        )
    return trainloader


def get_local_tokens_pt(
    pt_path: str,
    tokenizer: AutoTokenizer,  # kept for signature consistency; not used
    max_sequence_length: int,
    num_calibration_samples: Optional[int] = None,
    seed: int = 42
) -> List[torch.Tensor]:
    """
    Load a local .pt file saved by your make_calib script:
      torch.save({"input_ids": tensor[n, seq_len], ...}, pt_path)
    and return a list of (1, max_sequence_length) tensors.
    This is the most stable/fast path.
    """
    obj = torch.load(pt_path, map_location="cpu")
    if isinstance(obj, dict) and "input_ids" in obj:
        input_ids = obj["input_ids"]
    else:
        input_ids = obj

    if not isinstance(input_ids, torch.Tensor):
        raise ValueError(f"Unsupported .pt format: expected Tensor or dict with 'input_ids', got {type(input_ids)}")

    if input_ids.dim() != 2:
        raise ValueError(f"Expected input_ids shape [N, L], got {tuple(input_ids.shape)}")

    N, L = input_ids.shape
    if L < max_sequence_length:
        raise ValueError(f"Sequence length in pt ({L}) < required ({max_sequence_length})")

    # If pt has longer sequences, we take a random window per row.
    # If exactly equal, we take as-is.
    if num_calibration_samples is None:
        num_calibration_samples = N
    num_calibration_samples = min(num_calibration_samples, N)

    # Ensure long dtype for models
    input_ids = input_ids.to(dtype=torch.long)

    trainloader: List[torch.Tensor] = []
    for idx in range(num_calibration_samples):
        row = input_ids[idx]
        if L == max_sequence_length:
            chunk = row
        else:
            start = random.randint(0, L - max_sequence_length)
            chunk = row[start:start + max_sequence_length]
        trainloader.append(chunk.unsqueeze(0))  # (1, seq_len)

    return trainloader


def get_data(
    dataset_name: str, 
    tokenizer: AutoTokenizer, 
    max_sequence_length: int,
    num_calibration_samples: Optional[int] = None,
    seed: int = 42
) -> List[torch.Tensor]:
   
    # ---- NEW: local file support (does not affect existing dataset names) ----
    if os.path.isfile(dataset_name):
        lower = dataset_name.lower()
        if lower.endswith((".jsonl", ".json")):
            return get_local_jsonl(dataset_name, tokenizer, max_sequence_length, num_calibration_samples, seed)
        if lower.endswith(".pt"):
            return get_local_tokens_pt(dataset_name, tokenizer, max_sequence_length, num_calibration_samples, seed)

    # ---- NEW: allow full HF name alias for fineweb-edu ----
    if dataset_name in ("fineweb-edu", "HuggingFaceFW/fineweb-edu"):
        return get_fineweb_edu(tokenizer, max_sequence_length, num_calibration_samples, seed)


    if dataset_name == "open-thoughts":
        return get_open_thoughts(tokenizer, max_sequence_length, num_calibration_samples, seed)
    if dataset_name == "open-platypus":
        return get_open_platypus(tokenizer, max_sequence_length, num_calibration_samples, seed)
    if dataset_name == "ultrachat-200k":
        return get_ultrachat_200k(tokenizer, max_sequence_length, num_calibration_samples, seed)
    if dataset_name == "fineweb-edu":
        return get_fineweb_edu(tokenizer, max_sequence_length, num_calibration_samples, seed)
    if dataset_name == "c4":
        return get_c4(tokenizer, max_sequence_length, num_calibration_samples, seed)
    if dataset_name == "tulu":
        return get_tulu3_sft_mixture(tokenizer, max_sequence_length, num_calibration_samples, seed)
    else:
        raise ValueError("Unknown dataset")
