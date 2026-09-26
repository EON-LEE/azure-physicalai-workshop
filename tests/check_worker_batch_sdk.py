"""Run with the locked worker Python; no pytest, credentials or Azure calls required."""

import json
import sys
from importlib.metadata import version


def main():
    attempts = []

    def offline_only(event, _args):
        if event in {
            "socket.connect",
            "socket.getaddrinfo",
            "socket.gethostbyname",
            "socket.sendto",
            "subprocess.Popen",
        }:
            attempts.append(event)
            raise RuntimeError("SDK import must not contact a network or launch a process.")

    sys.addaudithook(offline_only)
    from azure.batch import BatchClient

    assert version("azure-batch") == "15.1.0"
    assert BatchClient.__name__ == "BatchClient"
    assert not {"isaacsim", "omni", "torch", "carb", "pxr", "lerobot", "transformers"} & set(
        sys.modules
    )
    assert attempts == []
    print(
        json.dumps(
            {
                "azure_batch": version("azure-batch"),
                "batch_client": f"{BatchClient.__module__}.{BatchClient.__name__}",
                "network_or_process_attempts": attempts,
                "gpu_modules_loaded": [],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
