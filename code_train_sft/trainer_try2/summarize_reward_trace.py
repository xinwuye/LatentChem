import argparse
import csv
import glob
import json
import os
from collections import defaultdict


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize per-rollout GRPO reward traces grouped by task/subtask."
    )
    parser.add_argument(
        "--trace_dir",
        type=str,
        required=True,
        help="Directory containing rank_*.jsonl files produced by --log_reward_trace true.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Optional CSV path for the aggregated summary.",
    )
    return parser.parse_args()


def _load_trace_rows(trace_dir: str):
    pattern = os.path.join(trace_dir, "rank_*.jsonl")
    trace_files = sorted(glob.glob(pattern))
    if not trace_files:
        raise FileNotFoundError(f"No reward trace files matched: {pattern}")

    rows = []
    expected_reward_names = None
    for trace_file in trace_files:
        with open(trace_file, "r", encoding="utf-8") as handle:
            for line_num, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "task" not in row or "subtask" not in row or "subrewards" not in row:
                    raise ValueError(
                        f"Trace row is missing required keys in {trace_file}:{line_num}: {row}"
                    )
                reward_names = tuple(row["subrewards"].keys())
                if expected_reward_names is None:
                    expected_reward_names = reward_names
                elif reward_names != expected_reward_names:
                    raise ValueError(
                        "Inconsistent reward names across trace rows. "
                        f"Expected {expected_reward_names}, got {reward_names} at {trace_file}:{line_num}."
                    )
                rows.append(row)

    if expected_reward_names is None:
        raise ValueError(f"Reward trace directory contains no non-empty JSONL rows: {trace_dir}")
    return trace_files, rows, list(expected_reward_names)


def _init_group_stats(reward_names):
    stats = {
        "count": 0,
        "reward_total_raw_sum": 0.0,
        "reward_total_raw_count": 0,
        "subreward_sum": {name: 0.0 for name in reward_names},
        "subreward_count": {name: 0 for name in reward_names},
        "subreward_hit": {name: 0 for name in reward_names},
    }
    return stats


def _safe_mean(total: float, count: int):
    if count == 0:
        return None
    return total / count


def _aggregate_rows(rows, reward_names):
    grouped = defaultdict(lambda: _init_group_stats(reward_names))
    overall = _init_group_stats(reward_names)

    for row in rows:
        key = (row["task"], row["subtask"])
        reward_total_raw = row.get("reward_total_raw")
        subrewards = row["subrewards"]

        for target in (grouped[key], overall):
            target["count"] += 1
            if reward_total_raw is not None:
                target["reward_total_raw_sum"] += float(reward_total_raw)
                target["reward_total_raw_count"] += 1
            for reward_name in reward_names:
                value = subrewards.get(reward_name)
                if value is None:
                    continue
                value_f = float(value)
                target["subreward_sum"][reward_name] += value_f
                target["subreward_count"][reward_name] += 1
                if value_f > 0.0:
                    target["subreward_hit"][reward_name] += 1

    summary_rows = []
    for (task, subtask), stats in sorted(grouped.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))):
        summary_rows.append(_build_summary_row(task, subtask, stats, reward_names))
    summary_rows.append(_build_summary_row("__all__", "__all__", overall, reward_names))
    return summary_rows


def _build_summary_row(task, subtask, stats, reward_names):
    row = {
        "task": task,
        "subtask": subtask,
        "count": stats["count"],
        "reward_total_raw_mean": _safe_mean(stats["reward_total_raw_sum"], stats["reward_total_raw_count"]),
    }
    for reward_name in reward_names:
        mean_key = f"{reward_name}__mean"
        hit_key = f"{reward_name}__hit_rate"
        row[mean_key] = _safe_mean(stats["subreward_sum"][reward_name], stats["subreward_count"][reward_name])
        row[hit_key] = _safe_mean(stats["subreward_hit"][reward_name], stats["subreward_count"][reward_name])
    return row


def _write_csv(rows, output_csv):
    if not rows:
        raise ValueError("No summary rows to write.")
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(output_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main():
    args = _parse_args()
    trace_dir = os.path.abspath(args.trace_dir)
    if not os.path.isdir(trace_dir):
        raise FileNotFoundError(f"Trace directory does not exist: {trace_dir}")

    trace_files, rows, reward_names = _load_trace_rows(trace_dir)
    summary_rows = _aggregate_rows(rows, reward_names)

    print(f"trace_dir\t{trace_dir}")
    print(f"trace_files\t{len(trace_files)}")
    print(f"trace_rows\t{len(rows)}")
    print("")

    header = list(summary_rows[0].keys())
    print("\t".join(header))
    for row in summary_rows:
        print("\t".join(_format_value(row[col]) for col in header))

    if args.output_csv:
        _write_csv(summary_rows, args.output_csv)
        print("")
        print(f"wrote_csv\t{os.path.abspath(args.output_csv)}")


if __name__ == "__main__":
    main()
