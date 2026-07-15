#!/usr/bin/env python3
import argparse
import math
import os
import random

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision.utils import save_image
from tqdm import tqdm

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from bf_eval.benchmark_utils import create_or_load_fixed_split, write_csv, write_json
from bf_eval.metrics import compute_clip_i_values, compute_fid_from_paths, evaluate_full


PROJECT_ROOT = "/share/home/u2515283058/BF-Fashion-"
MYMODEL_ROOT = "/share/home/u2515283058/Mymodel"
DATA_ROOT = "/share/home/u2515283058/datasets/BF/training"

METRIC_FIELDS = [
    "ssim",
    "clip_i_real",
    "clip_i_texture",
    "tcf_lab_delta",
    "tcf_hsv_l1",
    "tcf_rgb_l2",
    "tpf_patch_sim",
    "tpf_gram_l1",
    "leak_colored_frac",
    "leak_mean_saturation",
    "leak_value_shift",
    "leak_edge_density",
    "struct_edge_f1",
    "struct_iou",
    "struct_edge_l1",
]

DISPLAY_METRICS = [
    ("FID ↓", "FID", None, ".4f"),
    ("CLIP-I Real ↑", "clip_i_real_mean", "clip_i_real_std", ".4f"),
    ("CLIP-I Texture ↑", "clip_i_texture_mean", "clip_i_texture_std", ".4f"),
    ("SSIM ↑", "ssim_mean", "ssim_std", ".4f"),
    ("TCF-LAB ↓", "tcf_lab_delta_mean", "tcf_lab_delta_std", ".4f"),
    ("TCF-HSV ↓", "tcf_hsv_l1_mean", "tcf_hsv_l1_std", ".4f"),
    ("TCF-RGB-L2 ↓", "tcf_rgb_l2_mean", "tcf_rgb_l2_std", ".4f"),
    ("TPF-Patch ↑", "tpf_patch_sim_mean", "tpf_patch_sim_std", ".4f"),
    ("TPF-Gram ↓", "tpf_gram_l1_mean", "tpf_gram_l1_std", ".4f"),
    ("Leak Colored ↓", "leak_colored_frac_mean", "leak_colored_frac_std", ".4f"),
    (
        "Leak Saturation ↓",
        "leak_mean_saturation_mean",
        "leak_mean_saturation_std",
        ".4f",
    ),
    ("Leak Value Shift ↓", "leak_value_shift_mean", "leak_value_shift_std", ".4f"),
    (
        "Boundary Edge Density ↓",
        "leak_edge_density_mean",
        "leak_edge_density_std",
        ".4f",
    ),
    ("Edge F1 ↑", "struct_edge_f1_mean", "struct_edge_f1_std", ".4f"),
    ("Sketch IoU ↑", "struct_iou_mean", "struct_iou_std", ".4f"),
    ("Edge L1 ↓", "struct_edge_l1_mean", "struct_edge_l1_std", ".4f"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate 500 BF-Fashion images and compute one global metric summary."
    )
    parser.add_argument("--config", default="configs/latent-diffusion/mydata_ldm-vq-f8_test.yaml")
    parser.add_argument("--ckpt", default="last.ckpt")
    parser.add_argument(
        "--dataset-json",
        default=os.path.join(MYMODEL_ROOT, "data", "train_bf_texture.json"),
    )
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument(
        "--split-path",
        default=os.path.join(
            MYMODEL_ROOT, "eval", "benchmarks", "phase1_bf_val_split.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(PROJECT_ROOT, "eval_output", "benchmark_500_total_metrics"),
    )
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generation-seed", type=int, default=42)
    parser.add_argument("--ddim-steps", type=int, default=200)
    parser.add_argument("--ddim-eta", type=float, default=0.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fid-batch-size", type=int, default=16)
    parser.add_argument("--clip-batch-size", type=int, default=16)
    parser.add_argument(
        "--clip-model-path",
        default=os.path.join(MYMODEL_ROOT, "models", "clip"),
    )
    parser.add_argument("--min-valid-pixels", type=int, default=50)
    parser.add_argument("--metrics-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(data_root, value):
    if not value:
        return None
    return value if os.path.isabs(value) else os.path.join(data_root, value)


def load_model(config, ckpt, device):
    print(f"Loading BF-Fashion model from {ckpt}")
    checkpoint = torch.load(ckpt, map_location="cpu")
    model = instantiate_from_config(config.model)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.to(device)
    model.eval()
    return model


def image_tensor(path, mode, device):
    image = np.asarray(Image.open(path).convert(mode), dtype=np.uint8)
    if image.ndim == 2:
        image = image[:, :, None]
    image = np.transpose(image, (2, 0, 1))
    image = (image / 127.5 - 1.0).astype(np.float32)
    return torch.from_numpy(image[None]).to(device)


def set_generation_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def generate_one(model, sampler, config, sketch_path, texture_path, args, sample_id):
    set_generation_seed(args.generation_seed + int(sample_id))
    sketch = image_tensor(sketch_path, "L", args.device)
    texture = image_tensor(texture_path, "RGB", args.device)
    conditioning = model.get_learned_conditioning(
        {"sketch": sketch, "texture": texture}
    )
    unconditional_conditioning = {
        "two_zero": model.get_learned_conditioning(
            {
                "sketch": torch.zeros_like(sketch),
                "texture": torch.zeros_like(texture),
            }
        ),
        "one_zero": model.get_learned_conditioning(
            {"sketch": torch.zeros_like(sketch), "texture": texture}
        ),
    }
    latent_size = int(config.model.params.image_size)
    samples, _ = sampler.sample(
        S=args.ddim_steps,
        conditioning={"cond": conditioning},
        batch_size=1,
        shape=[4, latent_size, latent_size],
        verbose=False,
        unconditional_guidance_scale={
            "scale_1": config.model.params.scale_1,
            "scale_2": config.model.params.scale_2,
        },
        unconditional_conditioning=unconditional_conditioning,
        eta=args.ddim_eta,
    )
    decoded = model.decode_first_stage(samples)
    return torch.clamp((decoded + 1.0) / 2.0, 0.0, 1.0)


def finite_values(rows, key):
    return [
        float(row[key])
        for row in rows
        if isinstance(row.get(key), (int, float, np.number))
        and math.isfinite(float(row[key]))
    ]


def aggregate_metrics(rows, fid_value, fid_backend):
    summary = {
        "scope": "ALL",
        "count": len(rows),
        "FID": float(fid_value),
        "FID_std": None,
        "FID_backend": fid_backend,
    }
    for key in METRIC_FIELDS:
        values = finite_values(rows, key)
        summary[f"{key}_mean"] = float(np.mean(values)) if values else None
        summary[f"{key}_std"] = float(np.std(values)) if values else None
        summary[f"{key}_valid"] = len(values)
    return summary


def format_summary(summary):
    lines = ["BF-Fashion 500-image total metrics", "=" * 48]
    for label, key, std_key, number_format in DISPLAY_METRICS:
        value = summary.get(key)
        rendered = "N/A" if value is None else format(value, number_format)
        std_value = summary.get(std_key) if std_key else None
        if value is not None and std_value is not None:
            rendered += f" ± {format(std_value, number_format)}"
        lines.append(f"{label}: {rendered}")
    lines.extend(
        [
            "=" * 48,
            f"Samples: {summary['count']}",
            f"FID backend: {summary['FID_backend']}",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    generated_dir = os.path.join(args.output_dir, "generated")
    os.makedirs(generated_dir, exist_ok=True)

    split = create_or_load_fixed_split(
        args.dataset_json,
        args.split_path,
        num_samples=args.num_samples,
        seed=args.seed,
        sample_id_start=0,
        sample_id_end=args.num_samples,
    )
    write_json(os.path.join(args.output_dir, "benchmark_samples.json"), split)

    rows = []
    for sample in split:
        sample_id = sample["sample_id"]
        rows.append(
            {
                "sample_id": sample_id,
                "dataset_index": sample["idx"],
                "gen_path": os.path.join(generated_dir, f"{sample_id}.png"),
                "target_path": resolve_path(args.data_root, sample.get("target")),
                "texture_path": resolve_path(args.data_root, sample.get("texture")),
                "sketch_path": resolve_path(args.data_root, sample.get("sketch")),
                "mask_path": resolve_path(args.data_root, sample.get("mask")),
            }
        )

    for key in ("target_path", "texture_path", "sketch_path"):
        missing_inputs = [
            row[key] for row in rows if not row[key] or not os.path.isfile(row[key])
        ]
        if missing_inputs:
            raise RuntimeError(f"missing {key}: {len(missing_inputs)}/{len(rows)}")

    if not args.metrics_only:
        config = OmegaConf.load(args.config)
        model = load_model(config, args.ckpt, args.device)
        sampler = DDIMSampler(model)
        for sample, row in tqdm(
            zip(split, rows), total=len(rows), desc="Generating"
        ):
            if os.path.isfile(row["gen_path"]) and not args.overwrite:
                continue
            generated = generate_one(
                model,
                sampler,
                config,
                row["sketch_path"],
                row["texture_path"],
                args,
                sample["sample_id"],
            )
            save_image(generated, row["gen_path"])
        del sampler, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    missing = [row["gen_path"] for row in rows if not os.path.isfile(row["gen_path"])]
    if missing:
        raise RuntimeError(
            f"generated images are incomplete: {len(rows) - len(missing)}/{len(rows)}"
        )

    for row in tqdm(rows, desc="Per-image metrics"):
        metrics = evaluate_full(
            row["gen_path"],
            target_path=row["target_path"],
            texture_path=row["texture_path"],
            sketch_path=row["sketch_path"],
            mask_path=row["mask_path"],
            compute_leakage=True,
            compute_structure=True,
            min_valid_pixels=args.min_valid_pixels,
        )
        row.update(metrics)

    gen_paths = [row["gen_path"] for row in rows]
    real_paths = [row["target_path"] for row in rows]
    texture_paths = [row["texture_path"] for row in rows]

    print("Computing CLIP-I Real...")
    clip_real = compute_clip_i_values(
        gen_paths,
        real_paths,
        batch_size=args.clip_batch_size,
        device=args.device,
        model_name=args.clip_model_path,
    )
    print("Computing CLIP-I Texture...")
    clip_texture = compute_clip_i_values(
        gen_paths,
        texture_paths,
        batch_size=args.clip_batch_size,
        device=args.device,
        model_name=args.clip_model_path,
    )
    for row, real_value, texture_value in zip(rows, clip_real, clip_texture):
        row["clip_i_real"] = float(real_value)
        row["clip_i_texture"] = float(texture_value)

    print("Computing one global FID over all 500 images...")
    fid_value, fid_backend = compute_fid_from_paths(
        gen_paths,
        real_paths,
        batch_size=args.fid_batch_size,
        device=args.device,
        return_backend=True,
    )
    summary = aggregate_metrics(rows, fid_value, fid_backend)

    write_csv(os.path.join(args.output_dir, "metrics_per_sample.csv"), rows)
    write_json(os.path.join(args.output_dir, "metrics_per_sample.json"), rows)
    write_csv(os.path.join(args.output_dir, "metrics_summary.csv"), [summary])
    write_json(os.path.join(args.output_dir, "metrics_summary.json"), [summary])

    summary_text = format_summary(summary)
    with open(
        os.path.join(args.output_dir, "total_metrics.txt"), "w", encoding="utf-8"
    ) as handle:
        handle.write(summary_text)
    print("\n" + summary_text)
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
