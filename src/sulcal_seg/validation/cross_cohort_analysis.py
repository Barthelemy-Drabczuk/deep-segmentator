"""Cross-cohort generalisation analysis (stub)."""
from typing import Any, Dict, List


class CrossCohortAnalyzer:
    """
    Analyses model performance across multiple cohorts
    (UKBioBank, ABCD, ABIDE, SENIOR) to assess generalisation.

    This is a stub — it requires pre-computed per-subject metrics.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def analyse(
        self,
        cohort_metrics: Dict[str, List[Dict[str, float]]],
    ) -> Dict[str, Any]:
        """
        Aggregate and compare metrics across cohorts.

        Args:
            cohort_metrics: Dict mapping cohort name → list of per-subject metric dicts.

        Returns:
            Summary with per-cohort means, cross-cohort variance, and ranking.

        Raises:
            NotImplementedError: Always (stub).
        """
        # TODO: Compute per-cohort mean/std for each metric,
        #   run statistical tests (e.g., Kruskal-Wallis),
        #   flag cohorts with significantly lower performance.
        raise NotImplementedError(
            "CrossCohortAnalyzer.analyse() is not yet implemented."
        )
