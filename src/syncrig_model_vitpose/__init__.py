"""ViTPose provider for SyncRig.

Pip-installable plug-in adding the ``vitpose`` provider — transformer-
based 2D body pose (COCO_17 keypoints) via the HuggingFace
``usyd-community/vitpose-base-simple`` weights, paired with a
torchvision FasterRCNN person detector (engine-shared, BSD-3).

Externalised from public SyncRig because the 2D-only output is
considered niche; MediaPipe Holistic covers most users at a lower cost.
Apache 2.0 throughout — no commercial restrictions.
"""
from __future__ import annotations

from .provider import VitPoseProvider  # noqa: F401  side-effect: ProviderRegistry

__version__ = "0.1.0"
__all__ = ["VitPoseProvider"]
