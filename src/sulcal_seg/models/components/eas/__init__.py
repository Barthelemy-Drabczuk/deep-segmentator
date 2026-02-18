"""EAS (Evolutionary Architecture Search) component."""
# Config import deferred to avoid Pydantic version conflicts at import time.
# Import directly: from sulcal_seg.models.components.eas.config import EASComponentConfig
from .eas_engine import EASEngine
from .fitness_eval import FitnessEvaluator
from .hosvd_breeder import HOSVDBreeder
from .mutation_ops import change_activation, change_depth, change_filters, mutate, tournament_select
from .search_space import ArchitectureGenome, SearchSpace

__all__ = [
    "EASEngine",
    "FitnessEvaluator",
    "HOSVDBreeder",
    "ArchitectureGenome",
    "SearchSpace",
    "mutate",
    "change_filters",
    "change_activation",
    "change_depth",
    "tournament_select",
]
