"""CLI entrypoint for ai-dev."""

import typer

app = typer.Typer(
    name="ai-dev",
    help="AI Dev Toolkit CLI for evaluating LLM responses.",
    no_args_is_help=True,
)


@app.command()
def version() -> None:
    """Show the toolkit version."""
    from ai_dev_toolkit import __version__

    typer.echo(f"ai-dev-toolkit v{__version__}")


if __name__ == "__main__":
    app()
