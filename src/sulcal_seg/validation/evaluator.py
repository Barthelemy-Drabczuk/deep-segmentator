"""Model evaluator stub."""
from typing import Any, Dict


class ModelEvaluator:
    """
    Evaluates a trained SulcalSegmentationModel on a dataset.

    This is a stub — it requires GPU and real data loaders.
    """

    def __init__(self, model: Any, config: Dict[str, Any]) -> None:
        self.model = model
        self.config = config

    def evaluate_dataset(
        self,
        loader: Any,
        num_classes: int = 41,
    ) -> Dict[str, float]:
        """
        Run inference on all samples in `loader` and compute validation metrics.

        Args:
            loader: DataLoader providing image/label pairs.
            num_classes: Number of segmentation classes.

        Returns:
            Dict mapping metric name to aggregate value.

        Raises:
            NotImplementedError: Always (stub — requires GPU + data).
        """
        # TODO: Iterate loader, run model.forward(), accumulate metrics
        #   via sulcal_seg.validation.metrics.compute_all_metrics().
        raise NotImplementedError(
            "ModelEvaluator.evaluate_dataset() requires GPU and real data loaders."
        )
