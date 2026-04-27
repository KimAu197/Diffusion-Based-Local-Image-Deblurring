# Dataset Format

Each local deblurring sample contains:

- `Ib`: locally blurred RGB input image.
- `M`: blur mask as a single-channel image. Nonzero pixels mark the region to restore.
- `S`: optional segmentation or object map aligned to `Ib`.
- `target`: sharp RGB target used for training and evaluation.

Manifest files are JSON lists or objects with a `samples` list:

```json
{
  "samples": [
    {
      "sample_id": "example_000",
      "Ib": "example_000/Ib.png",
      "M": "example_000/M.png",
      "S": "example_000/S.png",
      "target": "example_000/target.png"
    }
  ]
}
```

Paths may be absolute or relative to the manifest directory.

## Source Assumptions

COCO-style preparation can use instance masks as object-aware blur masks. Global blur data can be adapted by generating arbitrary masks and compositing blurred regions back into sharp images. ReLoBlur-style fine-tuning should provide paired local blur inputs, masks, and sharp targets.

Use `python scripts/prepare_synthetic_data.py --dry-run` to create generated fixtures under `output/` before using real datasets. Full dataset-scale preparation requires user confirmation.
