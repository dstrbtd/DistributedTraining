import json
import os
import shutil
from influxdb_client import InfluxDBClient, Point, WritePrecision
from tabulate import tabulate

# json_file = "/root/llama_1b_7_49_0_2025_08_19T16_47_05/distributed__llama-1b/results_2025-08-19T14-07-09.202177.json"
# output_dir = "llama_1b_7_41_0_2025_08_19T12_30_19"
output_dirs = [
    "llama_1b_7_49_0_2025_08_19T16_47_05",
    "llama_1b_7_50_0_2025_08_19T17_39_35",
    "llama_1b_7_51_0_2025_08_19T18_32_12",
    "llama_1b_7_52_0_2025_08_20T00_24_22",
    "llama_1b_7_53_0_2025_08_20T01_15_43",
    "llama_1b_7_54_0_2025_08_20T02_06_28",
    "llama_1b_7_55_0_2025_08_20T07_58_08",
    "llama_1b_7_56_0_2025_08_20T08_50_16",
    "llama_1b_7_57_0_2025_08_20T09_42_33",
]
for output_dir in output_dirs:
    REPO_ID = "distributed/llama-1b"

    directory = f"{os.getcwd()}/{output_dir}/{REPO_ID.replace('/', '__')}"
    json_file = f"{directory}/{os.listdir(directory)[0]}"
    new_output_dir = f"{os.path.dirname(os.path.abspath(__file__))}/{output_dir}.json"

    tag = "7.58.0"

    # Load JSON
    with open(json_file, "r") as f:
        data = json.load(f)

    timestamp = int(data.get("date", 0) * 1e9)  # Influx expects ns

    for task, values in data["results"].items():
        alias = values.get("alias", task)
        for metric, score in values.items():
            if metric == "alias":
                continue  # skip alias itself
            else:
                print(task + "." + metric.replace(",none", ""), score)
            try:
                score = float(score)

                point = (
                    Point("eval_scores")  # measurement
                    .tag("tag", tag)
                    .tag("task", f"task+'.'+metric.replace(',none', '')")
                    .field("value", score)
                    .time(timestamp, WritePrecision.NS)
                )

                # write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=point)
            except (TypeError, ValueError):
                continue  # skip non-numeric

    # ---- PRINT SUMMARY TABLE ----
    results = data["results"]

    def get_score(task, metric):
        val = results.get(task, {}).get(metric, None)
        return f"{val*100:.1f}" if isinstance(val, float) else "N/A"  # convert to %

    # Collect rows
    rows = [
        [
            "DSTRBTD-1B",
            "FineWebEdu",
            "100B",
            get_score("hellaswag", "acc_norm,none"),
            get_score("piqa", "acc_norm,none"),
            get_score("arc_easy", "acc,none"),
        ],
        [
            "TEMPLAR-1B",
            "FineWebEdu",
            "100B-200B",
            51.0,
            71.4,
            59.2,
        ],
        [
            "DEM0-1B",
            "Dolmo",
            "100B",
            "48.0",
            "70.0",
            "55.0",
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
