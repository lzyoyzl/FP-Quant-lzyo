import os, json, argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="本地模型目录，用来取 tokenizer")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_seqs", type=int, default=1024)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--config", default="sample-10BT")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    # 有些 Llama tokenizer 没显式 pad，校准不需要 pad；这里不强制设置。

    # streaming: 不会把巨型数据集下载到本地，只拉取你迭代到的部分
    ds = load_dataset(args.dataset, args.config, split=args.split, streaming=True)

    # 通过拼接 token 来严格得到 n_seqs * seq_len
    needed = args.n_seqs * args.seq_len
    buf = []
    total = 0

    # 为了可复现：固定读取顺序（streaming 本身是确定顺序的），seed 这里主要是记录
    for ex in ds:
        text = ex.get("text", "")
        if not text:
            continue
        ids = tok(text, add_special_tokens=False).input_ids
        if not ids:
            continue
        buf.extend(ids)
        total = len(buf)
        if total >= needed:
            break

    if len(buf) < needed:
        raise RuntimeError(f"Not enough tokens collected: {len(buf)} < {needed}. "
                           f"Try iterating longer or change dataset/config.")

    buf = buf[:needed]
    input_ids = torch.tensor(buf, dtype=torch.int32).view(args.n_seqs, args.seq_len)

    # 1) 严格 token 版
    pt_path = os.path.join(args.out_dir, f"fineweb_calib_{args.n_seqs}x{args.seq_len}_tokens.pt")
    torch.save({"input_ids": input_ids, "seed": args.seed}, pt_path)

    # 2) 文本 jsonl 版（将每个 block decode 回文本，便于肉眼检查/备用）
    jsonl_path = os.path.join(args.out_dir, f"fineweb_calib_{args.n_seqs}x{args.seq_len}_text.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for i in range(args.n_seqs):
            txt = tok.decode(input_ids[i].tolist(), skip_special_tokens=True)
            f.write(json.dumps({"text": txt}, ensure_ascii=False) + "\n")

    print("Saved:")
    print("  ", pt_path)
    print("  ", jsonl_path)

if __name__ == "__main__":
    main()
