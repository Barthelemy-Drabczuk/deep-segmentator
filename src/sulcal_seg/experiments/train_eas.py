"""CLI script: Train the EAS component (evolutionary architecture search)."""
import click
import yaml

from sulcal_seg.models.builder import ModelBuilder


@click.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to YAML config file")
@click.option("--output-dir", "-o", default="outputs/eas", show_default=True, help="Output directory for checkpoints")
@click.option("--device", default="cuda", show_default=True, help="Device: cuda or cpu")
def main(config: str, output_dir: str, device: str) -> None:
    """Run the EAS evolutionary architecture search."""
    with open(config) as f:
        cfg = yaml.safe_load(f)

    eas_cfg = cfg.get("eas", cfg)
    model = ModelBuilder().with_eas(eas_cfg).build()
    click.echo(f"Built: {model}")

    # TODO: Load data loaders from cfg, call model.eas.train(train_loader, val_loader)
    raise NotImplementedError("train_eas: training loop requires GPU and real data.")


if __name__ == "__main__":
    main()
