"""CLI script: Train the full EAS + GAN + Laplace model."""
import click
import yaml

from sulcal_seg.models.builder import ModelBuilder


@click.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to YAML config file")
@click.option("--output-dir", "-o", default="outputs/full", show_default=True, help="Output directory")
@click.option("--device", default="cuda", show_default=True, help="Device: cuda or cpu")
@click.option("--num-epochs", default=None, type=int, help="Override number of epochs from config")
def main(config: str, output_dir: str, device: str, num_epochs: int) -> None:
    """Train the full sulcal segmentation model (EAS + GAN + Laplace)."""
    with open(config) as f:
        cfg = yaml.safe_load(f)

    model = (
        ModelBuilder()
        .with_eas(cfg["eas"])
        .with_gan(cfg["gan"])
        .with_laplace(cfg["laplace"])
        .build()
    )
    click.echo(f"Built: {model}")
    click.echo(f"Summary: {model.component_summary()}")

    # TODO: Load data, build Trainer, call trainer.train()
    raise NotImplementedError("train_full: training loop requires GPU and real data.")


if __name__ == "__main__":
    main()
