"""Execute the experiment with this Python environment and visible progress."""
from pathlib import Path
import sys
import time

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpec


def main():
    root = Path(__file__).resolve().parents[1]
    path = root / "notebooks" / "rag_experiments.ipynb"
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    started = time.monotonic()

    def progress(cell, cell_index, **kwargs):
        if cell.cell_type == "code":
            print(f"Cell {cell_index + 1}/{len(notebook.cells)}: "
                  f"{cell.source.splitlines()[0]}", flush=True)

    client = NotebookClient(
        notebook, timeout=None, allow_errors=False,
        resources={"metadata": {"path": str(root)}},
        on_cell_start=progress,
    )
    manager = KernelManager()
    manager._kernel_spec = KernelSpec(
        argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        display_name="Project virtual environment", language="python",
    )
    client.km = manager
    with client.setup_kernel():
        for index, cell in enumerate(notebook.cells):
            client.execute_cell(cell, index)
            if cell.cell_type == "code":
                print(f"Completed in {time.monotonic() - started:.0f}s total", flush=True)
                for output in cell.outputs:
                    if output.output_type == "stream":
                        print(output.text, end="", flush=True)
    nbformat.validate(notebook)
    with path.open('w', encoding='utf-8', newline='\n') as destination:
        nbformat.write(notebook, destination)
    print(f"Saved executed notebook: {path}", flush=True)


if __name__ == "__main__":
    main()
