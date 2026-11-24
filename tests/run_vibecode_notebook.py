"""Execute only vibecode-test1 with the existing project kernel; embed real outputs.

Uses installed jupyter_client, not nbclient/nbformat. No dependency installation.
Run from the project root using .venv/Scripts/python.exe.
"""
import json
import os
from pathlib import Path
from queue import Empty
import sys
import tempfile
import time

from jupyter_client import KernelManager


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "notebooks/vibecode-test1.ipynb"


def main():
    notebook = json.loads(PATH.read_text(encoding="utf-8"))
    runtime = tempfile.TemporaryDirectory(prefix="vibecode-kernel-")
    manager = KernelManager(kernel_name="python3", connection_file=str(Path(runtime.name) / "kernel.json"))
    manager.kernel_spec.argv[0] = sys.executable
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    environment["IPYTHONDIR"] = str(Path(runtime.name) / "ipython")
    environment["MPLCONFIGDIR"] = str(Path(runtime.name) / "matplotlib")
    manager.start_kernel(cwd=str(ROOT), env=environment,
                         extra_arguments=["--HistoryManager.hist_file=:memory:"])
    client = manager.client()
    client.start_channels()
    try:
        client.wait_for_ready(timeout=90)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            cell["outputs"] = []
            cell["execution_count"] = None
            source = "".join(cell["source"])
            started = time.monotonic()
            message_id = client.execute(source, stop_on_error=True)
            failed = False
            while True:
                if time.monotonic() - started > 300:
                    raise TimeoutError(f"Cell {index}: exceeded 300 seconds")
                try:
                    message = client.get_iopub_msg(timeout=1)
                except Empty:
                    continue
                if message["parent_header"].get("msg_id") != message_id:
                    continue
                kind, content = message["msg_type"], message["content"]
                if kind == "execute_input":
                    cell["execution_count"] = content["execution_count"]
                elif kind == "stream":
                    cell["outputs"].append(dict(output_type="stream", name=content["name"], text=content["text"]))
                elif kind in ("display_data", "execute_result"):
                    output = dict(output_type=kind, data=content["data"], metadata=content.get("metadata", {}))
                    if kind == "execute_result":
                        output["execution_count"] = content["execution_count"]
                    cell["outputs"].append(output)
                elif kind == "error":
                    failed = True
                    cell["outputs"].append(dict(output_type="error", ename=content["ename"],
                                                evalue=content["evalue"], traceback=content["traceback"]))
                elif kind == "clear_output" and not content.get("wait"):
                    cell["outputs"] = []
                elif kind == "status" and content["execution_state"] == "idle":
                    break
            # Save after every cell: useful outputs survive a later failure.
            PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"Cell {index:02d}: {'ERROR' if failed else 'OK'} ({time.monotonic()-started:.1f}s), "
                  f"{len(cell['outputs'])} outputs", flush=True)
            if failed:
                error = next(o for o in cell["outputs"] if o["output_type"] == "error")
                raise RuntimeError(f"Cell {index}: {error['ename']}: {error['evalue']}")
        images = sum("image/png" in o.get("data", {}) for c in notebook["cells"]
                     for o in c.get("outputs", []))
        print(f"Completed notebook with {images} embedded PNG figures: {PATH}")
    finally:
        manager.shutdown_kernel(now=True)
        client.stop_channels()
        runtime.cleanup()


if __name__ == "__main__":
    main()
