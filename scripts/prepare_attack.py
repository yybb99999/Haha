"""Relocate verified attack configs and public dataset split paths."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}; select a new --run_id.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def prepare(run_id, device="cuda", output_root=None):
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_id):
        raise ValueError("run_id must contain letters, digits, hyphens or underscores only")
    runtime = ROOT / "attack/configs/local" / run_id
    if runtime.exists():
        raise FileExistsError(f"{runtime} already exists; select a new run_id")
    splits = ROOT / "attack/splits/paper"
    manifest = json.loads((splits / "split_manifest.template.json").read_text(encoding="utf-8"))
    for key, value in list(manifest.items()):
        if key.endswith("_path"):
            manifest[key] = str(splits / value)
        elif key.endswith("_paths"):
            manifest[key] = [str(splits / name) for name in value]
    for key, value in manifest.items():
        paths = [value] if key.endswith("_path") else value if key.endswith("_paths") else []
        for path in paths:
            if not Path(path).is_file():
                raise FileNotFoundError(path)
    outputs = Path(output_root).resolve() if output_root else ROOT / "outputs/attack"
    prepared = []
    for template in sorted((ROOT / "attack/configs/templates").glob("*.json")):
        config = json.loads(template.read_text(encoding="utf-8"))
        cert = ROOT / "attack/configs/certificates" / config.pop("certificate_file")
        expected = config.pop("certificate_sha256")
        if hashlib.sha256(cert.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Certificate hash mismatch: {cert}")
        config.update({"project_root": str(ROOT / "attack"), "python_exe": sys.executable,
            "privacy_certificate_path": str(cert),
            "split_manifest_path": str(runtime / "split_manifest.json"),
            "output_dir": str(outputs / run_id / template.stem),
            "run_tag": run_id + "_" + template.stem})
        config["train"]["device"] = device
        config["export"]["device"] = device
        prepared.append((runtime / template.name, config))
    write_json(runtime / "split_manifest.json", manifest)
    for path, config in prepared:
        write_json(path, config)
    return [path for path, _ in prepared]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_id", default="reproduction01")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output_root", type=Path)
    args = parser.parse_args()
    for path in prepare(args.run_id, args.device, args.output_root):
        print(path)
    print("Prepared paths only; no training launched. Run pipelines from the attack directory.")


if __name__ == "__main__":
    main()
