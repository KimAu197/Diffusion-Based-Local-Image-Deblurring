# Data Status

## Found Locally

- COCO2017: `/autodl-pub/data/COCO2017`
  - Validation zip: `/autodl-pub/data/COCO2017/val2017.zip`
  - Training zip: `/autodl-pub/data/COCO2017/train2017.zip`
  - Annotations zip: `/autodl-pub/data/COCO2017/annotations_trainval2017.zip`
- COCO2014: `/autodl-pub/data/COCO14`
  - Training zip: `/autodl-pub/data/COCO14/train2014.zip`
  - Validation zip: `/autodl-pub/data/COCO14/val2014.zip`

## Not Found Locally

- GoPro: public source is `https://huggingface.co/datasets/snah/GOPRO_Large`
- RealBlur: public source is `http://cg.postech.ac.kr/research/RealBlur/`
- ReLoBlur: public source is `https://github.com/LeiaLi/ReLoBlur`
  - Official train: Google Drive folder `1rAPKzhhRjztj7Utbb00BJLSVaPC-1Jua`; Baidu code `49nb`
  - Official test: Google Drive folder `1nYj4e7TSXeqBsUZxLvoay_JLZ7wxdNmC`; Baidu code `nmcy`
  - Official masks: Google Drive folder `1-4YerKKlDydgoBeZbiV0_XR9iJLKbLXI`; Baidu code `98mw`
  - License/access: academic use, CC BY-NC-SA 4.0 on the project page.
  - Current machine status: Google Drive access failed with `Network is unreachable`; no extracted ReLoBlur image/mask files were found under `/root`, `/root/autodl-tmp`, or `/autodl-pub/data`.
  - Baidu status: official shares are reachable and extract-code verification works. Listings were saved under `/root/autodl-tmp/reloblur_raw/baidu_listing_{test,train,masks}.json`.
    - Test share: `test.zip`, 3,267,130,190 bytes.
    - Train share: `s1-4.zip`, `s5-8.zip`, `s9-11.zip`, `s12-15.zip`, `s16.zip`, `s17-19.zip`, 15,398,217,837 bytes total.
    - Mask share: `test.zip`, `train.zip`, 61,343,879 bytes total.
  - Baidu download blocker: direct PCS download returns `user not exists`; BaiduPCS-Go transfer returns `获取分享项元数据错误`; `api/sharedownload` requires page signing/verification. A Baidu authenticated download method or valid browser cookies are needed to fetch the archives.
  - Converter ready: after data is placed locally, run `python scripts/prepare_reloblur_manifest.py --dataset-root <path>/dataset --masks-root <path>/masks --output-dir output/datasets/reloblur --split all --validate-images`.

## Processed

- COCO2017 grouped semantic local blur dataset:
  - Manifest: `output/datasets/coco2017_train_grouped_localblur_5k/manifest.json`
  - Samples: 5000
  - Image size: 512 x 512
  - Fields: `Ib`, `M`, `S`, `target`
  - Blur variants: motion, Gaussian, defocus
  - Source: `/autodl-pub/data/COCO2017/train2017.zip`
  - Annotation source: `/autodl-pub/data/COCO2017/annotations_trainval2017.zip`
  - Selection rule: prefer moving foreground categories such as person, car, motorcycle, bicycle, dog, bird, truck, bus, train, horse, skateboard, and sports ball.
  - Grouping rule: person instances are merged with nearby carried or attached categories such as backpack, handbag, umbrella, tie, suitcase, cell phone, skis, snowboard, skateboard, and sports ball.
  - Quality rule: edge instances and black-border crops are filtered; this existing dataset constrained final mask area to 2% to 35% of the crop.
  - Current generator default: new synthetic COCO data is constrained to 5% to 20% mask area, uses milder default blur parameters, and records per-sample blur parameters in the manifest metadata.
  - Stats: `output/datasets/coco2017_train_grouped_localblur_5k/dataset_stats.json`

- Deprecated COCO2017 semantic local blur dataset:
  - Manifest: `output/datasets/coco2017_val_semantic_localblur_5k/manifest.json`
  - Status: deprecated because it was generated before grouped motion-object rules were added.
