"""Run with locked worker Python and the repository on PYTHONPATH; no Azure calls required."""

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

    from apps.learning_worker import candidate_provenance, managed_reports
    from simulation import batch, batch_learned, batch_task, paired_evaluation

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
                "controller_modules": [
                    batch.__name__,
                    batch_task.__name__,
                    batch_learned.__name__,
                    paired_evaluation.__name__,
                ],
                "consumer_modules": [managed_reports.__name__, candidate_provenance.__name__],
                "network_or_process_attempts": attempts,
                "gpu_modules_loaded": [],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
