import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _arg_value(command: list[str], name: str) -> str | None:
    if name not in command:
        return None
    return command[command.index(name) + 1]


def replace_arg(command: list[str], name: str, values: list[str]) -> list[str]:
    values = [str(v) for v in values]
    if name not in command:
        return [*command, name, *values]
    i = command.index(name)
    end = i + 2
    while end < len(command) and not command[end].startswith("--"):
        end += 1
    return [*command[: i + 1], *values, *command[end:]]


def output_path(command: list[str]) -> Path | None:
    out_dir = _arg_value(command, "--output_dir")
    if out_dir is None:
        return None
    if "merging_experiments/fisher_merging_demo.py" in command:
        dataset = _arg_value(command, "--dataset") or "mnist"
        dataset_tag = "" if dataset == "mnist" else f"{dataset}_"
        batch_size_a = _arg_value(command, "--batch_size_a")
        batch_size_b = _arg_value(command, "--batch_size_b")
        examples = _arg_value(command, "--examples_per_model")
        shard_size_a = _arg_value(command, "--shard_size_a")
        steps = _arg_value(command, "--steps") or "2000"
        seed = _arg_value(command, "--seed") or "0"
        if None in (batch_size_a, batch_size_b):
            return None
        budget_tag = f"e{examples}" if examples is not None else f"s{steps}"
        shard_tag = f"_na{shard_size_a}" if shard_size_a is not None else ""
        return Path(out_dir) / (
            f"merge_{dataset_tag}m{batch_size_a}v{batch_size_b}_{budget_tag}{shard_tag}_seed{seed}.json"
        )
    opt = _arg_value(command, "--optimizer_name")
    beta2 = _arg_value(command, "--beta2")
    bs = _arg_value(command, "--batch_size")
    rid = _arg_value(command, "--run_id")
    if None in (opt, beta2, bs, rid):
        return None
    return Path(out_dir) / f"{opt}_{beta2}_{bs}_{rid}.json"


def prepare(command: list[str], device: str) -> list[str]:
    cmd = [sys.executable if p == "python" else p for p in command]
    return replace_arg(cmd, "--device", [device])


def run_one(index: int, command: list[str]) -> dict:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return {
        "index": index,
        "returncode": result.returncode,
        "command": command,
        "tail": result.stdout[-4000:],
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run a sweep plan with resumability and optional parallelism."
    )
    p.add_argument("--plan", required=True, help="Path to sweep plan JSON.")
    p.add_argument("--workers", type=int, default=1, help="Concurrent subprocesses.")
    p.add_argument("--device", default="cuda", help="Device passed to each run.")
    p.add_argument("--dry-run", action="store_true", help="Print commands without running.")
    args = p.parse_args()

    with open(args.plan) as f:
        commands: list[list[str]] = json.load(f)["commands"]

    skipped: list[int] = []
    to_run: list[tuple[int, list[str]]] = []
    for i, cmd in enumerate(commands):
        out = output_path(cmd)
        if out is not None and out.exists():
            skipped.append(i)
        else:
            to_run.append((i, prepare(cmd, args.device)))

    total = len(commands)
    print(f"Plan: {total} total, {len(skipped)} already done, {len(to_run)} to run.")

    if args.dry_run:
        for i, cmd in to_run:
            print(f"[{i}] {' '.join(cmd)}")
        return

    done = len(skipped)
    results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_one, i, cmd): i for i, cmd in to_run}
        for fut in concurrent.futures.as_completed(futures):
            r = fut.result()
            results.append(r)
            done += 1
            status = "ok" if r["returncode"] == 0 else f"FAILED:{r['returncode']}"
            print(f"[{done}/{total}] {status} (job {r['index']})", flush=True)

    failed = [r for r in results if r["returncode"] != 0]
    if failed:
        print(f"\n{len(failed)} run(s) failed:")
        for r in sorted(failed, key=lambda x: x["index"]):
            print(f"  [{r['index']}]", " ".join(r["command"]))
            print(r["tail"][-1000:])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
