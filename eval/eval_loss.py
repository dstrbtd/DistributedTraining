import asyncio
import time
import random
import datetime
from zoneinfo import ZoneInfo
from typing import Callable, Dict

from dotenv import load_dotenv
import os
import gc
import torch
import shutil
import json
import torch.distributed as dist
from distributed_training import __run__, __spec_version__
from distributed_training.data.dataset import DatasetLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE
from pathlib import Path
from influxdb_client import InfluxDBClient, Point, WritePrecision
from tabulate import tabulate

load_dotenv()
# === CONFIG ===
INFLUXDB_URL = os.getenv("INFLUXDB_URL")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET")
INFLUXDB_MEASUREMENT = "evaluation_metrics"
REPO_ID = "dstrbtd/llama-1b"
BUCKET = "llama-4b-ws-4"
DATASET_ID = "HuggingFaceFW/fineweb-edu"
DATASET_SKIP_PROBABILITY = 0.9
EVAL_DURATION_MINUTES = 30
EVAL_TYPES = ["fineweb", "lm_eval"]  # Add lm-eval harness tasks in the future
__run__ = "11"

# === INFLUXDB SETUP ===
influx = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
write_api = influx.write_api()
query_api = influx.query_api()
delete_api = influx.delete_api()


def tag_exists(tag: str, task: str) -> bool:
    if task == "lm_eval":
        task = "mmlu_stem.acc"
    query = f"""
    from(bucket: "{INFLUXDB_BUCKET}")
    |> range(start: -365d)
    |> filter(fn: (r) => r._measurement == "{INFLUXDB_MEASUREMENT}" and r.tag == "{tag}" and r.task == "{task}" and r.spec_version == "{__spec_version__}")
    |> limit(n:1)
    """
    result = query_api.query(org=INFLUXDB_ORG, query=query)
    return len(result) > 0


# query = f'''from(bucket: "{INFLUXDB_BUCKET}") |> range(start: -365d) |> filter(fn: (r) => r._measurement == "{INFLUXDB_MEASUREMENT}" and r.tag == "{tag}" and r.task == "{task}") |> limit(n:1)'''
# query = f'''from(bucket: "{INFLUXDB_BUCKET}") |> range(start: -365d) |> filter(fn: (r) => r._measurement == "{INFLUXDB_MEASUREMENT}")'''
# delete_api.delete('1970-01-01T00:00:00Z', '2025-08-11T00:00:00Z', f'_measurement="{INFLUXDB_MEASUREMENT}"', bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG)


def log_score(
    tag: str,
    task: str,
    score: float,
    output_dir: str = None,
):
    if task == "fineweb":
        point = (
            Point(INFLUXDB_MEASUREMENT)
            .tag("tag", tag)
            .tag("task", task)
            .tag("spec_version", __spec_version__)
            .field("score", score)
            .time(datetime.datetime.now(datetime.timezone.utc), WritePrecision.NS)
        )
        write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=point)
    else:
        directory = f"{os.getcwd()}/{output_dir}/{REPO_ID.replace('/', '__')}"
        json_file = f"{directory}/{os.listdir(directory)[0]}"
        new_output_dir = (
            f"{os.path.dirname(os.path.abspath(__file__))}/{output_dir}.json"
        )

        # Load JSON
        with open(json_file, "r") as f:
            data = json.load(f)

        timestamp = int(data.get("date", 0) * 1e9)  # Influx expects ns
        for task, values in data["results"].items():
            for metric, score in values.items():
                if metric == "alias":
                    continue  # skip alias itself
                # else:
                #     print(task+"."+metric.replace(",none", ""), score)
                try:
                    score = float(score)
                    point = (
                        Point(INFLUXDB_MEASUREMENT)  # measurement
                        .tag("tag", tag)
                        .tag("task", f"{task}.{metric.replace(',none', '')}")
                        .tag("spec_version", __spec_version__)
                        .field("score", score)
                        .time(timestamp, WritePrecision.NS)
                    )

                    write_api.write(
                        bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=point
                    )

                except (TypeError, ValueError) as e:
                    print(f"An error occurred: {e}")
                    continue  # skip non-numeric

        # ---- PRINT SUMMARY TABLE ----
        results = data["results"]

        def get_score(task, metric):
            val = results.get(task, {}).get(metric, None)
            return f"{val*100:.1f}" if isinstance(val, float) else "N/A"  # convert to %

        # Collect rows
        rows = [
            [
                "DSTRBTD-1.10B",
                "FineWebEdu",
                f'~{int(80*int(tag.split(".")[1])*100*512*1025/1e9)}B',  # 32 peers per outer step, 65 outer steps, 100 inner steps, 512 samples per inner_step, 1024 Tokens Per Sample
                get_score("hellaswag", "acc_norm,none"),
                get_score("piqa", "acc_norm,none"),
                get_score("arc_easy", "acc,none"),
            ],
            [
                "TEMPLAR-1.21B",
                "FineWebEdu",
                "100B-200B",
                51.0,
                71.4,
                59.2,
            ],
            [
                "DEM0-1.18B",
                "Dolmo",
                "100B",
                "48.0",
                "71.0",
                "55.0",
            ],
            [
                "DILOCO-1.30B",
                "Dolmo",
                "26B",
                "45.0",
                "68.4",
                "39.0",
            ],
        ]

        print(
            tabulate(
                rows,
                headers=[
                    "Model",
                    "Dataset",
                    "Tokens",
                    "HellaSwag acc_norm",
                    "PIQA acc_norm",
                    "ARC-E acc",
                ],
                tablefmt="fancy_grid",
            )
        )

        shutil.rmtree(f"{os.getcwd()}/{output_dir}")
        with open(new_output_dir, "w") as f:
            json.dump(data, f, indent=4)  # indent=4 for pretty printing


async def fetch_training_data(tokenizer):
    """Async function to fetch training data"""
    retry_limit = 10
    retry_delay = 60
    attempt = 0
    current_block = random.randint(6193881 * 2, 6193881 * 4)
    uid = random.randint(300, 1000000)
    local_batch_size_train = 4
    while attempt < retry_limit:
        try:
            pages = await DatasetLoader.next_pages(
                offset=current_block,
                n_pages=35,
                seed=uid,
            )
            random.seed(uid)
            random.shuffle(pages)

            dataset = await DatasetLoader.create(
                batch_size=local_batch_size_train,
                sequence_length=1024,
                pages_info=pages,
                tokenizer=tokenizer,
            )

            return dataset
        except Exception as e:
            print(f"Error fetching training data: {str(e)}")
            attempt += 1
            print(f"Failed to fetch data, retrying. Attempt {attempt}/{retry_limit}")
            if attempt < retry_limit:
                time.sleep(retry_delay * attempt)  # Wait before the next retry
            else:
                print("Maximum retry limit reached. Unable to fetch data.")
                raise


# === EVALUATORS ===
def evaluate_fineweb(
    device: str,
    tag: str,
    max_minutes: int = EVAL_DURATION_MINUTES,
) -> float:
    """
    Stream and evaluate a fixed-time sample of fineweb-edu on average LM loss.

    Args:
        model: HuggingFace model
        tokenizer: Matching tokenizer
        device: cuda or cpu
        max_minutes: Time budget in minutes
        max_seq_length: Max input length for tokenization

    Returns:
        Average loss
    """
    print(f"[⏳] Downloading model for tag {tag}...")
    model_path = snapshot_download(repo_id=REPO_ID, revision=tag)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    ).to(device)

    model.eval()
    total_loss = 0.0
    n_batches = 0
    start_time = time.time()

    loop = asyncio.new_event_loop()

    while (time.time() - start_time) <= (max_minutes * 60):
        # Use streaming mode
        dataset = loop.run_until_complete(fetch_training_data(tokenizer))

        with torch.no_grad():
            for i, batch in enumerate(dataset):
                # breakpoint()
                if random.random() > (1 - DATASET_SKIP_PROBABILITY):
                    continue

                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                if inputs is None or len(inputs) == 0:
                    print(f"Empty batch at index {i}, skipping")
                    continue

                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    outputs = model(input_ids=inputs, labels=inputs)
                    total_loss += outputs.loss.item()
                    n_batches += 1
                    # print(total_loss/n_batches)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    score = total_loss / n_batches if n_batches > 0 else float("inf")
    log_score(tag, "fineweb", score)
    return score


def evaluate_with_lm_harness(
    device: str,
    tag: str,
) -> float:
    """
    Evaluate model using lm-eval-harness (e.g. HellaSwag, ARC).
    """
    output_dir = f"{BUCKET.replace('-', '_')}_{tag.replace('.','_')}_{datetime.datetime.now(ZoneInfo('Africa/Cairo')).strftime('%Y_%m_%dT%H_%M_%S')}"
    tasks = [
        "hellaswag",
        "arc_challenge",
        "arc_easy",
        "openbookqa",
        "winogrande",
        "piqa",
        "mmlu",
    ]

    cmd_parts = [
        "lm-eval",
        "--model hf",
        f"--model_args pretrained=/root/{BUCKET},tokenizer={REPO_ID},parallelize=True",
        f"--tasks {','.join(tasks)}",
        # f"--device {device}",
        f"--batch_size 256",
        f"--output_path {output_dir}",
    ]

    # command = " ".join(cmd_parts) + " >/dev/null 2>&1"
    command = " ".join(cmd_parts)
    start_time = time.time()
    print(f"Running command: {command}")
    exit_code = os.system(command)
    score = 0
    # exit_code = 0
    # breakpoint()
    if exit_code == 0:
        log_score(tag, "lm_eval", score, output_dir)
    # breakpoint()
    benchmark_runtime = time.time() - start_time
    # breakpoint()
    return score


# === EVALUATION REGISTRY ===


def get_evaluator(task: str) -> Callable:
    if task == "fineweb":
        return evaluate_fineweb
    elif task in ["hellaswag", "arc_easy", "arc_challenge", "lm_eval"]:
        return evaluate_with_lm_harness
    else:
        raise ValueError(f"Unsupported evaluation task: {task}")


# === MAIN LOOP ===


def evaluate_all_tags_once():
    api = HfApi()
    refs = api.list_repo_refs(REPO_ID)
    tags = sorted(
        refs.tags, key=lambda p: (int(p.name.split(".")[0]), int(p.name.split(".")[1]))
    )
    sorted(
        refs.tags, key=lambda p: (int(p.name.split(".")[0]), int(p.name.split(".")[1]))
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for tag_obj in tags:
        tag = tag_obj.name
        try:
            if tag.split(".")[0] != __run__:
                continue
            else:
                print(f"\n=== [TAG] {tag} ===")

            # if int(tag.split(".")[1]) < 113:
            #     continue
            # if (int(tag.split(".")[1]) != 115) and (int(tag.split(".")[1]) != 120) and (int(tag.split(".")[1]) != 125) and (int(tag.split(".")[1]) != 130) and (int(tag.split(".")[1]) != 135):
            # if (int(tag.split(".")[1]) != 134):
            #     continue

            for task in EVAL_TYPES:
                # if tag_exists(tag, "fineweb"):
                if tag_exists(tag, task):
                    print(f"[✓] {task}: already evaluated")
                    continue

                if task == "fineweb":
                    continue

                print(f"[⏳] Evaluating {task}...")
                evaluator = get_evaluator(task)
                score = evaluator(device, tag)
                print(f"[✅] {task}: {score:.4f}")

        except Exception as e:
            print(f"[⚠️] Error evaluating tag {tag}: {e}")

        finally:
            cache_dir = HF_HUB_CACHE
            cache_dir = Path(cache_dir).expanduser().resolve()
            for cache in cache_dir.iterdir():
                if os.path.isdir(cache):
                    shutil.rmtree(str(cache))


# === Optional Continuous Mode ===


def monitor_repo(poll_interval_sec: int = 18000):
    print("[🔁] Starting continuous monitoring...")
    while True:
        evaluate_all_tags_once()
        print(f"[⏳] Sleeping for {poll_interval_sec}s...")
        time.sleep(poll_interval_sec)


# === Entry ===

if __name__ == "__main__":
    # evaluate_all_tags_once()
    monitor_repo()
