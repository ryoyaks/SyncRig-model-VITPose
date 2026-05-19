"""ViTPose provider for SyncRig — COCO_17 body keypoints via HuggingFace ViTPose.

Ships as ``syncrig-model-vitpose``: a pip-installable plugin for the
SyncRig engine. Apache 2.0 throughout — no commercial restrictions.
Externalised from public SyncRig because the 2D-only output is
considered niche; MediaPipe Holistic covers most users at a lower cost.

Outputs 17 normalised landmarks (COCO order). Topology adaptation to
MEDIAPIPE_33 happens in the client, not here.

Pipeline: torchvision FasterRCNN person detection → ViTPose keypoint
inference on each bbox.
"""

from __future__ import annotations

from syncrig_core.providers.helpers import require_modules

# Raise at module-top when the runtime extras aren't installed. The
# error message points at the canonical install command; the engine's
# entry-point autoloader catches this and surfaces the plugin under
# ``ep:vitpose`` (unavailable) so the Extensions UI's
# ``FailedEntryPointCard`` tells the user to re-install via the git
# URL field (which auto-installs ``[runtime]`` extras as of v0.3).
require_modules(
    ("torch", "torchvision", "transformers", "PIL"),
    pkg="syncrig-model-vitpose",
    install_command="pip install 'syncrig-model-vitpose[runtime]'",
)

import logging
from typing import TYPE_CHECKING

import cv2

from syncrig_core.providers import (
    OutputKind,
    Provider,
    ProviderCapabilities,
    ProviderConfigField,
    ProviderOutput,
    ProviderRegistry,
)
from syncrig_core.providers.depth_filter import DepthFilter
from syncrig_core.skeleton import SkeletonTopology

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

log = logging.getLogger(__name__)


@ProviderRegistry.register
class VitPoseProvider(Provider):
    """ViTPose body keypoint estimation (COCO_17)."""

    @classmethod
    def capabilities(cls) -> ProviderCapabilities:
        return ProviderCapabilities(
            name="vitpose",
            description="ViTPose — transformer-based 2D pose. COCO_17 keypoints; depth synthesised from limb shortening.",
            skeleton_topology=SkeletonTopology.COCO_17,
            outputs=frozenset({OutputKind.SKELETON}),
            requires_gpu=True,  # CPU works but very slow
            # External pip package — no uv-sync extra. The git-URL
            # installer (v0.3 plugin SDK) auto-installs ``[runtime]``
            # extras so there's no separate Extensions install button
            # to wire up; ``install_steps()`` stays empty.
            requires_extra=None,
            fps_estimate=20,
            device_kinds=frozenset({"cuda", "cpu"}),
            min_vram_gb=2.0,
            commercial="safe",
            commercial_note=(
                "Apache 2.0 weights + code; person detector is torchvision "
                "FasterRCNN (BSD-3). Commercial use unrestricted."
            ),
            user_label="ViTPose · 2D pose, commercial-safe",
            user_tagline="Lightweight body skeleton, no mesh",
            # v0.3 plugin SDK metadata — surfaces on the Extensions
            # card header + inside the git-install confirm modal.
            version="0.2.0",
            repository_url="https://github.com/ryoyaks/SyncRig-model-VITPose",
            homepage="https://huggingface.co/usyd-community/vitpose-base-simple",
            author="@ryoyaks",
            license="Apache-2.0",
            tags=frozenset({"body", "skeleton", "2d", "transformer"}),
            config_schema=(
                ProviderConfigField(
                    name="depth",
                    label="Synthesize depth (Z) via limb-shortening",
                    type="bool",
                    default=False,
                ),
            ),
        )

    def __init__(self) -> None:
        self._detector = None
        self._processor = None
        self._model = None
        self._cfg: dict = {}
        self._depth_filter: DepthFilter | None = None

    def setup(self, config: dict | None = None) -> None:
        # Heavy imports deferred until selected.
        from transformers import (  # noqa: PLC0415
            AutoProcessor,
            VitPoseForPoseEstimation,
        )

        # PersonDetector is the engine-shared torchvision FasterRCNN
        # wrapper — provided by syncrig-engine so every body provider
        # gets the same person-bbox semantics without each shipping its
        # own copy.
        from syncrig_engine.providers._person_detector import PersonDetector  # noqa: PLC0415

        cfg = config or {}
        det_model = cfg.get("detector", "mobilenet_v3_320")
        model_id = cfg.get("model", "usyd-community/vitpose-base-simple")
        try:
            self._detector = PersonDetector(model_name=det_model)
            self._detector.setup()
            self._processor = AutoProcessor.from_pretrained(model_id)
            self._model = VitPoseForPoseEstimation.from_pretrained(model_id)
            log.info("ViTPose loaded: detector=%s model=%s", det_model, model_id)
        except Exception:  # pylint: disable=broad-except
            log.exception("Failed to initialise ViTPose")
            self._detector = None
            self._processor = None
            self._model = None
        self._cfg = cfg
        self._depth_filter = DepthFilter() if cfg.get("depth") else None

    def process(self, frame: "NDArray[np.uint8]") -> ProviderOutput | None:
        if self._model is None or self._detector is None or self._processor is None:
            return None

        # Lazy imports — only present after a successful setup() call.
        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        h, w = frame.shape[:2]
        try:
            detections = self._detector.detect(frame)
        except Exception:  # pylint: disable=broad-except
            log.exception("ViTPose detector failed")
            return None
        if not detections:
            return None

        # Highest-scoring person bbox.
        x1, y1, x2, y2, _ = detections[0]
        bbox = [x1, y1, x2, y2]

        # ViTPose wants RGB PIL.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        try:
            inputs = self._processor(images=pil_img, boxes=[[bbox]], return_tensors="pt")
            with torch.no_grad():
                outputs = self._model(**inputs)
            pose_results = self._processor.post_process_pose_estimation(
                outputs,
                boxes=[[bbox]],
                threshold=self._cfg.get("keypoint_threshold", 0.3),
            )
        except Exception:  # pylint: disable=broad-except
            log.exception("ViTPose inference failed")
            return None

        if not pose_results or len(pose_results[0]) == 0:
            return None

        kpts = pose_results[0][0]["keypoints"].cpu().numpy()  # (17, 2)
        scores = pose_results[0][0]["scores"].cpu().numpy()    # (17,)

        landmarks: list[list[float]] = []
        visibility: list[float] = []
        for i, kp in enumerate(kpts):
            landmarks.append([float(kp[0]) / w, float(kp[1]) / h, 0.0])
            visibility.append(float(scores[i]))

        out = ProviderOutput(skeleton_topology=SkeletonTopology.COCO_17)
        out.pose_landmarks = landmarks
        out.visibility = visibility
        if self._depth_filter is not None:
            out = self._depth_filter.apply(out)
        return out

    def close(self) -> None:
        if self._detector is not None:
            self._detector.close()
        self._detector = None
        self._processor = None
        self._model = None
        self._depth_filter = None
