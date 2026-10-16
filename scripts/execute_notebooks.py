"""Automated executor for presentation notebooks with static PNG rendering.

Executes notebooks using `nbclient` and sets Plotly's default renderer to 'png'
so all charts are embedded as base64 images that render natively on GitHub's viewer.
"""

from __future__ import annotations

import logging
from pathlib import Path

import nbformat
from nbclient import NotebookClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NOTEBOOKS = [
    Path("notebooks/02_model_evaluation.ipynb"),
    Path("notebooks/03_demand_prediction_dashboard.ipynb"),
    Path("notebooks/04_business_impact_simulation.ipynb"),
]


def execute_notebook(nb_path: Path) -> None:
    """Execute a single notebook and persist its outputs with embedded PNG charts."""
    logger.info("Executing %s...", nb_path.name)
    with open(nb_path, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Ensure pio.renderers.default = 'png' is configured in the first code cell
    first_code = next(c for c in nb.cells if c.cell_type == "code")
    if "pio.renderers.default = 'png'" not in first_code.source:
        first_code.source = (
            "import plotly.io as pio\npio.renderers.default = 'png'\n" + first_code.source
        )

    client = NotebookClient(
        nb,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": "notebooks/"}},
    )
    executed = client.execute()

    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(executed, f)

    pngs = sum(
        1
        for c in executed.cells
        if c.cell_type == "code"
        for out in c.get("outputs", [])
        if "image/png" in out.get("data", {})
    )
    logger.info("Saved %s with %d embedded PNG charts.", nb_path.name, pngs)


def main() -> None:
    """Execute all registered notebooks in the repository."""
    for nb in NOTEBOOKS:
        if nb.exists():
            execute_notebook(nb)
        else:
            logger.warning("Notebook not found: %s", nb)


if __name__ == "__main__":
    main()
