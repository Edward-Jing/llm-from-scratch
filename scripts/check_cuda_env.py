"""Print CUDA, PyTorch, and GPU information on the current node."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys


def run_command(args: list[str]) -> None:
    """Run a diagnostic command when it exists."""

    executable = args[0]
    if shutil.which(executable) is None:
        print(f"{executable}: not found")
        return

    print(f"\n$ {' '.join(args)}")
    completed = subprocess.run(args, check=False, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout.strip())
    if completed.stderr:
        print(completed.stderr.strip())


def main() -> None:
    """Print diagnostics useful before launching training on MSBC."""

    print(f"python={sys.version}")
    print(f"executable={sys.executable}")
    print(f"platform={platform.platform()}")

    run_command(["hostname"])
    run_command(["nvidia-smi"])
    run_command(["mycuda"])

    try:
        import torch
    except ImportError:
        print("torch: not installed")
        return

    print(f"\ntorch={torch.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        return

    print(f"cuda_device_count={torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        total_gb = props.total_memory / 1024**3
        capability = torch.cuda.get_device_capability(index)
        print(
            f"device_{index}={props.name}, memory_gb={total_gb:.1f}, "
            f"capability={capability}"
        )

    print(f"bf16_supported={torch.cuda.is_bf16_supported()}")


if __name__ == "__main__":
    main()
