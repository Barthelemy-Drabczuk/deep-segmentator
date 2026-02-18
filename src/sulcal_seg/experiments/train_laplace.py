"""CLI script: Run the Laplace geometry post-processing component."""
import click
import yaml

from sulcal_seg.models.builder import ModelBuilder


@click.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to YAML config file")
@click.option("--output-dir", "-o", default="outputs/laplace", show_default=True, help="Output directory")
def main(config: str, output_dir: str) -> None:
    """Apply Laplace-Beltrami post-processing to pre-computed segmentations."""
    with open(config) as f:
        cfg = yaml.safe_load(f)

    laplace_cfg = cfg.get("laplace", cfg)
    model = ModelBuilder().with_laplace(laplace_cfg).build()
    click.echo(f"Built: {model}")

    # TODO: Load segmentation outputs, call model.laplace.forward()
    raise NotImplementedError("train_laplace: requires pre-computed segmentations.")


if __name__ == "__main__":
    main()
