import asyncio
import time
from transformers import AutoTokenizer
from dataset import DatasetLoader as DatasetLoaderOld
from dataset_loader import DatasetLoader as DatasetLoaderNew

def ascii_bar(value, max_value, width=40):
    """Return a simple ASCII bar proportional to value/max_value."""
    filled = int(value / max_value * width)
    return "█" * filled + "-" * (width - filled)

async def run_loaders():
    batch_size = 4
    sequence_length = 1024
    tokenizer = AutoTokenizer.from_pretrained("dstrbtd/llama-1b", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    results = []

    start = time.time()
    pages = await DatasetLoaderOld.next_pages(offset=6649657, n_pages=35, seed=43)
    loaderOld = await DatasetLoaderOld.create(batch_size=batch_size,
                                              sequence_length=sequence_length,
                                              pages_info=pages,
                                              tokenizer=tokenizer)
    t_old = time.time() - start
    results.append(("LoaderOld", t_old, len(loaderOld.buffer)))

    start = time.time()
    loaderNew = DatasetLoaderNew(debug=False, randomness=True,
                                 sequence_length=sequence_length,
                                 tokenizer=tokenizer)
    await loaderNew.load_bucket_data_to_buffer(max_configs=3, max_rows_per_group=100)
    loaderNew.prepare_batches(batch_size=batch_size)
    t_new = time.time() - start
    results.append(("LoaderNew", t_new, len(loaderNew.buffer)))

    print("\n=== Loader Timing Comparison ===")
    max_time = max(t for _, t, _ in results)
    for name, t, buf in results:
        bar = ascii_bar(t, max_time)
        print(f"{name.ljust(10)} | {bar} {t:.2f}s ({buf} samples)")

asyncio.run(run_loaders())
