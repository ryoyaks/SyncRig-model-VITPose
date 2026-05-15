# syncrig-model-vitpose

ViTPose provider plug-in for [SyncRig](https://github.com/ryoyaks/SyncRig).
Adds transformer-based 2D body keypoint estimation (COCO_17, 17
keypoints per person) via the HuggingFace
`usyd-community/vitpose-base-simple` weights, paired with a
torchvision FasterRCNN person detector (engine-shared, BSD-3).

## Why a separate repo

Public SyncRig keeps the in-tree provider set focused on outputs that
the typical mocap user reaches for (MediaPipe Holistic, ROMP, SAM 3D
Body). ViTPose is 2D-only and synthesises depth via limb-shortening —
useful as a research baseline or as the body half of a composite
provider, but most users prefer MediaPipe for the same niche.

Apache 2.0 throughout (weights, code, person detector), so there's no
license reason for the externalisation — it's purely about keeping the
public surface area minimal.

## Install

```bash
uv pip install 'syncrig-model-vitpose[runtime]'
# (or)  pip install 'syncrig-model-vitpose[runtime]'
```

Weights for ViTPose itself + the FasterRCNN person detector auto-
download from HuggingFace / torchvision hub on first use. No manual
download step.

## How it plugs in

The package exposes one [`syncrig.providers`][ep] entry-point:

```toml
[project.entry-points."syncrig.providers"]
vitpose = "syncrig_model_vitpose.provider:VitPoseProvider"
```

On engine startup SyncRig calls
`ProviderRegistry.autoload_entry_points("syncrig.providers")` which
finds this entry-point and registers `VitPoseProvider`. The Extensions
page card appears with the engine-supplied description + commercial
chip (✓ safe).

The `coco_17` topology is already a built-in in `syncrig-core`, so no
topology registration is needed.

[ep]: https://packaging.python.org/en/latest/specifications/entry-points/

## Licensing

| Component | License | Commercial? |
|---|---|---|
| This package (provider code) | Apache 2.0 | ✅ Yes |
| ViTPose model + weights (HF) | Apache 2.0 | ✅ Yes |
| torchvision FasterRCNN (person detector) | BSD-3 | ✅ Yes |

No restrictions on use.

## Cite

If you use ViTPose in research:

```bibtex
@inproceedings{xu2022vitpose,
  title     = {{ViTPose}: Simple Vision Transformer Baselines for Human Pose Estimation},
  author    = {Xu, Yufei and Zhang, Jing and Zhang, Qiming and Tao, Dacheng},
  booktitle = {NeurIPS},
  year      = {2022},
}
```
