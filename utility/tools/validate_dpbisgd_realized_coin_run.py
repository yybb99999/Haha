"""Read-only validator for a completed realized-coin DP-BiSGD run."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from privacy_analysis.filters import PessimisticRealizedCoinPLDFilter  
from utils.realized_filter_integrity import verify_legacy_core  


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _close(left, right, tolerance=1e-10):
    return abs(float(left) - float(right)) <= tolerance * max(
        1.0,
        abs(float(left)),
        abs(float(right)),
    )


def validate_run(run_dir, require_passed=True):
    run_dir = Path(run_dir).resolve()
    command = _load_json(run_dir / "command.json")
    metrics = _load_json(run_dir / "metrics.json")
    manifest = _load_json(run_dir / "manifest.json")

    errors = []
    for name, expected in manifest.get("artifacts", {}).items():
        path = run_dir / name
        if not path.is_file():
            errors.append(f"Missing artifact: {name}")
        elif _sha256(path) != expected:
            errors.append(f"Artifact hash mismatch: {name}")

    if require_passed and metrics.get("status") != "passed":
        errors.append(f"Run status is not passed: {metrics.get('status')}")
    if metrics.get("status") != manifest.get("status"):
        errors.append("Manifest and metrics status differ.")
    if metrics.get("accounted_steps") != metrics.get("actual_dp_updates"):
        errors.append("Accounted steps and optimizer updates differ.")
    branch_total = (
        int(metrics.get("stepwise_small_steps", 0))
        + int(metrics.get("stepwise_large_steps", 0))
    )
    if branch_total != metrics.get("accounted_steps"):
        errors.append("Branch counts do not equal accounted steps.")

    pld_state = metrics["pld_state"]
    accountant = PessimisticRealizedCoinPLDFilter(
        target_epsilon=metrics["epsilon_target"],
        target_delta=metrics["delta_target"],
        sample_rate=metrics["sample_rate"],
        sigma_small=metrics["sigma_small"],
        sigma_large=metrics["sigma_large"],
        value_discretization_interval=(
            pld_state["value_discretization_interval"]
        ),
        log_mass_truncation_bound=pld_state["log_mass_truncation_bound"],
        tail_mass_truncation=pld_state["tail_mass_truncation"],
    )

    committed_events = []
    stop_events = []
    with (run_dir / "privacy_trace.jsonl").open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event") == "committed_update":
                committed_events.append(event)
                decision = accountant.preview(event["coin"]["branch"])
                if not decision.allowed:
                    errors.append(
                        f"Trace step {line_number} was committed above budget."
                    )
                    break
                if not _close(
                    decision.delta_upper,
                    event["pld_delta_upper"],
                ):
                    errors.append(
                        f"Trace PLD delta mismatch at line {line_number}."
                    )
                accountant.commit(decision.token)
            elif event.get("event") == "privacy_stop":
                stop_events.append(event)
                decision = accountant.preview(event["coin"]["branch"])
                if decision.allowed:
                    errors.append("Recorded stop branch is still within budget.")
                if not _close(
                    decision.delta_upper,
                    event["pld"]["delta_upper"],
                ):
                    errors.append("Recorded rejected PLD delta does not replay.")
            else:
                errors.append(f"Unknown trace event at line {line_number}.")

    if len(committed_events) != metrics.get("accounted_steps"):
        errors.append("Trace update count does not match metrics.")
    if metrics.get("status") == "passed" and len(stop_events) != 1:
        errors.append("A passed run must contain exactly one privacy stop event.")
    if not _close(
        accountant.current_delta_upper(),
        metrics["delta_upper_final_at_target_epsilon"],
    ):
        errors.append("Final PLD delta does not replay.")
    if accountant.current_delta_upper() > metrics["delta_target"]:
        errors.append("Final committed PLD exceeds the target delta.")

    with (run_dir / "result.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        errors.append("result.csv must contain exactly one data row.")
    elif int(rows[0]["Fin_Acc_T"]) != metrics["T_final"]:
        errors.append("result.csv Fin_Acc_T does not match T_final.")

    try:
        verified_hashes = verify_legacy_core(ROOT)
    except RuntimeError as exc:
        verified_hashes = None
        errors.append(str(exc))

    report = {
        "valid": not errors,
        "run_dir": str(run_dir),
        "status": metrics.get("status"),
        "stop_reason": metrics.get("stop_reason"),
        "T_final": metrics.get("T_final"),
        "small_steps": metrics.get("stepwise_small_steps"),
        "large_steps": metrics.get("stepwise_large_steps"),
        "final_delta_upper": accountant.current_delta_upper(),
        "target_delta": metrics.get("delta_target"),
        "legacy_sha256": verified_hashes,
        "errors": errors,
    }
    if errors:
        raise RuntimeError(json.dumps(report, indent=2, sort_keys=True))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--allow_incomplete", action="store_true")
    args = parser.parse_args()
    report = validate_run(
        args.run_dir,
        require_passed=not args.allow_incomplete,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
