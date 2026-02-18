"""CLI script: Evaluate a trained model on a validation dataset."""
import click
import yaml

from sulcal_seg.validation.evaluator import ModelEvaluator


@click.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to YAML config file")
@click.option("--checkpoint", "-k", required=True, type=click.Path(exists=True), help="Path to model checkpoint")
@click.option("--dataset", "-d", required=True, type=click.Choice(["ukbiobank", "abcd", "abide", "senior"]), help="Dataset to evaluate on")
@click.option("--output-dir", "-o", default="outputs/validation", show_default=True, help="Output directory for metrics")
def main(config: str, checkpoint: str, dataset: str, output_dir: str) -> None:
    """Evaluate a trained sulcal segmentation model."""
    with open(config) as f:
        cfg = yaml.safe_load(f)

    click.echo(f"Evaluating checkpoint: {checkpoint} on dataset: {dataset}")

    # TODO: Load model from checkpoint, create data loader, run ModelEvaluator
    raise NotImplementedError("validate_model: requires GPU and real checkpoint.")


if __name__ == "__main__":
    main()
