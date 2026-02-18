"""CLI script: Run all ablation study configurations."""
import click
import yaml

from sulcal_seg.models.builder import ModelBuilder


@click.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to YAML config file")
@click.option("--output-dir", "-o", default="outputs/ablations", show_default=True, help="Output directory")
@click.option("--study-name", default="ablation_study", show_default=True, help="Optuna study name (for HPO mode)")
@click.option("--storage", default=None, help="SQLite storage path (for HPO mode)")
@click.option("--n-trials", default=20, show_default=True, type=int, help="Number of HPO trials per configuration")
@click.option("--n-jobs", default=1, show_default=True, type=int, help="Parallel Optuna workers")
@click.option("--worker-id", default=0, show_default=True, type=int, help="SLURM array task ID")
def main(
    config: str,
    output_dir: str,
    study_name: str,
    storage: str,
    n_trials: int,
    n_jobs: int,
    worker_id: int,
) -> None:
    """
    Run all 7 ablation study configurations.

    Can operate in two modes:
        1. Grid mode: Build and evaluate each configuration once.
        2. HPO mode: Run Optuna optimisation for each configuration (requires --storage).
    """
    with open(config) as f:
        cfg = yaml.safe_load(f)

    click.echo(f"Building all ablation configurations...")
    ablations = ModelBuilder.build_all_ablations(
        eas_config=cfg["eas"],
        gan_config=cfg["gan"],
        laplace_config=cfg["laplace"],
    )
    for name, model in ablations.items():
        click.echo(f"  {name}: {model}")

    # TODO: For each ablation, train on GPU and collect metrics.
    raise NotImplementedError("run_ablations: training loop requires GPU and real data.")


if __name__ == "__main__":
    main()
