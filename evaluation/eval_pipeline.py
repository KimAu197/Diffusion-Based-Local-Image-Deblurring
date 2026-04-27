"""Evaluation loop entry point."""

from __future__ import annotations

import argparse

from local_deblur.config import load_yaml_config
from local_deblur.eval import build_eval_dataset, build_eval_model, create_eval_output, evaluate_sample, write_eval_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--round", required=True, dest="round_name")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--count", type=int, default=0)
    parser.add_argument("--mode", default="standard")
    parser.add_argument("--detailed", default="false")
    parser.add_argument("--config", default="configs/evaluation.yaml")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--visual-limit", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    evaluation_config = config.get("evaluation", {})
    data_config = config.get("data", {})
    model_config = config.get("model", {})
    split_config = config.get("split", {})
    manifest = args.manifest or data_config.get("manifest") or data_config.get("synthetic_manifest")
    checkpoint = args.checkpoint or model_config.get("checkpoint")
    split = args.split or split_config.get("name", evaluation_config.get("split", "val"))
    val_fraction = float(args.val_fraction if args.val_fraction is not None else split_config.get("val_fraction", 0.1))
    split_seed = int(args.split_seed if args.split_seed is not None else split_config.get("split_seed", 42))
    image_size = int(args.image_size if args.image_size is not None else evaluation_config.get("image_size", 512))

    output_dir, logger = create_eval_output(args.round_name, args.model, args.dataset, args.count)
    logger.info("starting evaluation args=%s", vars(args))
    logger.info(
        "resolved manifest=%s checkpoint=%s split=%s split_seed=%s val_fraction=%s image_size=%s dry_run=%s",
        manifest,
        checkpoint,
        split,
        split_seed,
        val_fraction,
        image_size,
        args.dry_run,
    )
    dataset = build_eval_dataset(
        args.dataset,
        args.count,
        dry_run=args.dry_run,
        manifest=manifest,
        split=split,
        val_fraction=val_fraction,
        split_seed=split_seed,
        image_size=image_size,
    )
    pipeline = build_eval_model(checkpoint=checkpoint, dry_run=args.dry_run, device=args.device)

    rows = []
    for sample in dataset:
        try:
            rows.append(evaluate_sample(sample, pipeline))
            logger.info("evaluated sample=%s", sample.sample_id)
        except Exception:
            logger.exception("failed sample=%s", getattr(sample, "sample_id", "unknown"))

    write_eval_outputs(
        output_dir,
        rows,
        model=args.model,
        dataset=args.dataset,
        dry_run=args.dry_run,
        checkpoint=checkpoint,
        split=split,
        manifest=manifest,
        visual_limit=args.visual_limit,
    )
    logger.info("wrote results to %s", output_dir)
    print(output_dir)


if __name__ == "__main__":
    main()
