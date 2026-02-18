"""CLI script: Train the GAN segmentation component."""
import click
import yaml

from sulcal_seg.models.builder import ModelBuilder


@click.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to YAML config file")
@click.option("--output-dir", "-o", default="outputs/gan", show_default=True, help="Output directory")
@click.option("--device", default="cuda", show_default=True, help="Device: cuda or cpu")
@click.option("--resume", default=None, help="Path to checkpoint to resume from")
def main(config: str, output_dir: str, device: str, resume: str) -> None:
    """Train the multi-label GAN segmentation component."""
    with open(config) as f:
        cfg = yaml.safe_load(f)

    gan_cfg = cfg.get("gan", cfg)
    model = ModelBuilder().with_gan(gan_cfg).build()
    click.echo(f"Built: {model}")

    # TODO: Load data loaders, build optimisers, call model.gan.train()
    raise NotImplementedError("train_gan: training loop requires GPU and real data.")


if __name__ == "__main__":
    main()
