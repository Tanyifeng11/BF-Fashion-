#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
import shutil
from pathlib import Path

import lpips
import torch
import torchvision.transforms as transforms
from PIL import Image
from pytorch_fid import fid_score
from tqdm import tqdm

from m_class_histo_loss import RGBuvHistBlock


PROJECT_DIR = "/share/home/u2515283058/BF-Fashion-"
DATASET_TEST_DIR = "/share/home/u2515283058/datasets/BF/test"
EVAL_OUTPUT_DIR = os.path.join(PROJECT_DIR, "eval_output")
DEFAULT_CLASSES = ["bag", "top", "outwear", "pants", "dress"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    parser.add_argument("--dataset-test-dir", default=DATASET_TEST_DIR)
    parser.add_argument("--eval-output", default=EVAL_OUTPUT_DIR)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(EVAL_OUTPUT_DIR, "benchmark_500_metrics"),
    )
    parser.add_argument("--split-path", default=None)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Total sample cap. Use -1 to keep per-class samples without global truncation.",
    )
    parser.add_argument("--per-class-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rebuild-split", type=int, choices=[0, 1], default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fid-batch-size", type=int, default=16)
    return parser.parse_args()


def image_files(folder):
    if not os.path.isdir(folder):
        return []
    return [
        name
        for name in sorted(os.listdir(folder))
        if Path(name).suffix.lower() in IMAGE_EXTS
    ]


def target_folder(class_folder):
    gt_folder = os.path.join(class_folder, "gt")
    if os.path.isdir(gt_folder):
        return gt_folder
    return os.path.join(class_folder, "cloth")


def paired_samples(args):
    rows = []
    for classname in args.classes:
        class_folder = os.path.join(args.dataset_test_dir, classname)
        real_dir = target_folder(class_folder)
        gen_dir = os.path.join(args.eval_output, "test", classname, "gen")
        real_names = set(image_files(real_dir))
        gen_names = set(image_files(gen_dir))
        names = sorted(real_names & gen_names)
        for name in names:
            rows.append(
                {
                    "class": classname,
                    "filename": name,
                    "real_path": os.path.join(real_dir, name),
                    "gen_path": os.path.join(gen_dir, name),
                }
            )
    return rows


def build_split(args):
    rng = random.Random(args.seed)
    rows = paired_samples(args)
    by_class = {classname: [] for classname in args.classes}
    for row in rows:
        by_class[row["class"]].append(row)

    selected = []
    for classname in args.classes:
        candidates = list(by_class[classname])
        rng.shuffle(candidates)
        selected.extend(candidates[: args.per_class_samples])

    if args.num_samples > 0 and len(selected) < args.num_samples:
        selected_keys = {(row["class"], row["filename"]) for row in selected}
        remaining = [
            row
            for row in rows
            if (row["class"], row["filename"]) not in selected_keys
        ]
        rng.shuffle(remaining)
        selected.extend(remaining[: args.num_samples - len(selected)])

    if args.num_samples > 0:
        selected = selected[: args.num_samples]
    selected.sort(key=lambda row: (row["class"], row["filename"]))

    expected = args.num_samples if args.num_samples > 0 else args.per_class_samples * len(args.classes)
    if len(selected) != expected:
        raise RuntimeError(
            f"only found {len(selected)} paired generated/real images, "
            f"expected {expected}"
        )
    return selected


def load_or_create_split(args):
    split_path = args.split_path or os.path.join(args.output_dir, "bf_500_split.json")
    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    if os.path.exists(split_path) and not args.rebuild_split:
        with open(split_path, "r", encoding="utf-8") as f:
            return json.load(f), split_path

    rows = build_split(args)
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    return rows, split_path


def reset_dir(path):
    os.makedirs(path, exist_ok=True)
    for item in Path(path).iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def link_or_copy(src, dst):
    try:
        os.symlink(src, dst)
    except OSError:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def prepare_subset_dirs(rows, output_dir, group_name):
    real_dir = os.path.join(output_dir, "subset_images", group_name, "real")
    gen_dir = os.path.join(output_dir, "subset_images", group_name, "gen")
    reset_dir(real_dir)
    reset_dir(gen_dir)
    for idx, row in enumerate(rows):
        suffix = Path(row["filename"]).suffix
        safe_name = f"{idx:05d}_{row['class']}{suffix}"
        link_or_copy(row["real_path"], os.path.join(real_dir, safe_name))
        link_or_copy(row["gen_path"], os.path.join(gen_dir, safe_name))
    return real_dir, gen_dir


def open_rgb(path):
    return Image.open(path).convert("RGB")


def compute_lpips(rows, device):
    transform = transforms.Compose([transforms.ToTensor()])
    loss_fn = lpips.LPIPS(net="alex").to(device)
    total = 0.0
    with torch.no_grad():
        for row in tqdm(rows, desc="LPIPS"):
            real = open_rgb(row["real_path"])
            gen = open_rgb(row["gen_path"])
            if real.size != gen.size:
                real = real.resize(gen.size, Image.BICUBIC)
            real_tensor = (2.0 * transform(real) - 1.0).unsqueeze(0).to(device)
            gen_tensor = (2.0 * transform(gen) - 1.0).unsqueeze(0).to(device)
            total += loss_fn(real_tensor, gen_tensor).item()
    return total / len(rows)


def image_tensor(path, device):
    image = transforms.ToTensor()(open_rgb(path)).unsqueeze(0)
    return image.to(device, torch.float)


def compute_histo(rows, device):
    block = RGBuvHistBlock(
        insz=256,
        h=64,
        intensity_scale=True,
        method="inverse-quadratic",
        hist_boundary=[-3, 3],
        device=device,
    )
    total = 0.0
    with torch.no_grad():
        for row in tqdm(rows, desc="Histo"):
            gen = image_tensor(row["gen_path"], device)
            real = image_tensor(row["real_path"], device)
            if real.shape[-2:] != gen.shape[-2:]:
                real = torch.nn.functional.interpolate(
                    real, size=gen.shape[-2:], mode="bilinear", align_corners=False
                )
            gen_hist = block(gen)
            real_hist = block(real)
            loss = (
                1
                / torch.sqrt(torch.tensor(2.0, device=device))
                * torch.sqrt(torch.sum(torch.pow(torch.sqrt(real_hist) - torch.sqrt(gen_hist), 2)))
                / gen_hist.shape[0]
            )
            total += loss.item()
    return total / len(rows)


def compute_fid(rows, args, group_name):
    real_dir, gen_dir = prepare_subset_dirs(rows, args.output_dir, group_name)
    return fid_score.calculate_fid_given_paths(
        [real_dir, gen_dir],
        batch_size=args.fid_batch_size,
        device=args.device,
        dims=2048,
    )


def metric_row(name, rows, args, device):
    print(f"[metrics] {name}: {len(rows)} samples")
    fid_value = compute_fid(rows, args, name)
    lpips_value = compute_lpips(rows, device)
    histo_value = compute_histo(rows, device)
    return {
        "class": name,
        "num_samples": len(rows),
        "fid": round(float(fid_value), 6),
        "lpips_alex": round(float(lpips_value), 6),
        "histo_loss": round(float(histo_value), 6),
    }


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    rows, split_path = load_or_create_split(args)
    manifest_csv = os.path.join(args.output_dir, "bf_500_split.csv")
    write_csv(manifest_csv, rows)

    grouped = {"ALL": rows}
    for classname in args.classes:
        grouped[classname] = [row for row in rows if row["class"] == classname]

    summary = []
    for name, group_rows in grouped.items():
        if not group_rows:
            continue
        summary.append(metric_row(name, group_rows, args, device))

    summary_csv = os.path.join(args.output_dir, "summary_metrics.csv")
    summary_json = os.path.join(args.output_dir, "summary_metrics.json")
    write_csv(summary_csv, summary)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(os.path.join(args.output_dir, "fid_lpips_500.txt"), "w", encoding="utf-8") as f:
        for row in summary:
            f.write(f"{row['class']},{row['fid']},{row['lpips_alex']}\n")
    with open(os.path.join(args.output_dir, "histo_500.txt"), "w", encoding="utf-8") as f:
        for row in summary:
            f.write(f"{row['class']}: {row['histo_loss']}\n")

    print("============================================")
    print("BF 500 metrics completed.")
    print(f"Split:   {split_path}")
    print(f"Summary: {summary_csv}")
    print("============================================")


if __name__ == "__main__":
    main()
