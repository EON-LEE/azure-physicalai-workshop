"""Safe exact RNG/data-order sidecars for the one-process pinned native training entry."""

from __future__ import annotations

import random
from pathlib import Path

from learning.common import canonical, digest, finite, integer, keys, read_json, require, write_json
from learning.smolvla.checkpoints import CONTINUATION_SCHEMA

DATA_SCHEMA = "physicalai.smolvla-data-order/v1"
MAX_DATASET_FRAMES = 1_000_000


def continuation_metadata(root: Path, expected_step: int) -> dict:
    value = keys(
        read_json(root / "training_state" / "continuation.json", 1024 * 1024),
        {
            "schema",
            "step",
            "sampler",
            "rng",
            "precision",
            "num_workers",
            "world_size",
            "mixed_precision",
        },
        "full-state continuation",
    )
    require(value["schema"] == CONTINUATION_SCHEMA, "Unknown exact continuation schema")
    integer(value["step"], "restored optimizer step", expected_step, expected_step)
    integer(value["num_workers"], "restored worker count", 0, 0)
    integer(value["world_size"], "restored world size", 1, 1)
    require(value["mixed_precision"] == "no", "AMP state is not an approved continuation mode")
    data = keys(
        value["sampler"],
        {
            "schema",
            "dataset_size",
            "indices_sha256",
            "batch_size",
            "epoch",
            "cursor",
            "batches",
        },
        "saved sampler metadata",
    )
    require(data["schema"] == DATA_SCHEMA, "Unknown data-order schema")
    integer(data["dataset_size"], "dataset frame count", 1, MAX_DATASET_FRAMES)
    integer(data["batch_size"], "saved batch size", 1, 64)
    integer(data["batches"], "saved consumed batches", expected_step, expected_step)
    integer(data["cursor"], "saved data cursor", 0, data["dataset_size"])
    integer(data["epoch"], "saved data epoch", 0, expected_step)
    from learning.common import sha256

    sha256(data["indices_sha256"], "saved dataset indices")
    rng = keys(
        value["rng"],
        {
            "python_version",
            "python_gaussian",
            "numpy_algorithm",
            "numpy_position",
            "numpy_has_gaussian",
            "numpy_gaussian",
            "cuda_device_count",
        },
        "saved RNG metadata",
    )
    require(
        rng["python_version"] == 3 and rng["numpy_algorithm"] == "MT19937",
        "Unknown saved Python/NumPy RNG algorithm",
    )
    if rng["python_gaussian"] is not None:
        finite(rng["python_gaussian"], "Python Gaussian cache")
    finite(rng["numpy_gaussian"], "NumPy Gaussian cache")
    integer(rng["numpy_position"], "NumPy RNG position", 0, 624)
    integer(rng["numpy_has_gaussian"], "NumPy cache flag", 0, 1)
    integer(rng["cuda_device_count"], "saved CUDA RNG count", 0, 1)
    precision = keys(
        value["precision"],
        {
            "matmul_precision",
            "matmul_tf32",
            "cudnn_tf32",
            "cudnn_benchmark",
            "cudnn_deterministic",
            "deterministic_algorithms",
            "default_dtype",
        },
        "saved precision flags",
    )
    require(
        precision["matmul_precision"] in ("highest", "high", "medium")
        and precision["default_dtype"]
        in ("torch.float32", "torch.float64", "torch.float16", "torch.bfloat16"),
        "Unknown saved floating-point configuration",
    )
    require(
        all(
            type(precision[name]) is bool
            for name in precision.keys() - {"matmul_precision", "default_dtype"}
        ),
        "Saved precision flags must be explicit booleans",
    )
    return value


def precision_state() -> dict:
    import torch

    return {
        "matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "default_dtype": str(torch.get_default_dtype()),
    }


def capture_rng() -> tuple[dict, dict]:
    import numpy as np
    import torch

    py, numpy = random.getstate(), np.random.get_state()
    require(numpy[0] == "MT19937" and py[0] == 3, "Unsupported exact RNG format")
    cuda_count = torch.cuda.device_count()
    require(cuda_count <= 1, "Full-state checkpoint supports one CUDA device only")
    metadata = {
        "python_version": py[0],
        "python_gaussian": py[2],
        "numpy_algorithm": numpy[0],
        "numpy_position": int(numpy[2]),
        "numpy_has_gaussian": int(numpy[3]),
        "numpy_gaussian": float(numpy[4]),
        "cuda_device_count": cuda_count,
    }
    tensors = {
        "python_mt": torch.tensor(py[1], dtype=torch.int64),
        "numpy_mt": torch.tensor(numpy[1].astype("int64"), dtype=torch.int64),
        "torch_rng": torch.get_rng_state().clone(),
    }
    if cuda_count:
        tensors["cuda_rng"] = torch.cuda.get_rng_state(0).clone()
    return metadata, tensors


def restore_rng(snapshot: tuple[dict, dict]) -> None:
    import numpy as np
    import torch

    metadata, tensors = snapshot
    keys(
        metadata,
        {
            "python_version",
            "python_gaussian",
            "numpy_algorithm",
            "numpy_position",
            "numpy_has_gaussian",
            "numpy_gaussian",
            "cuda_device_count",
        },
        "exact RNG metadata",
    )
    require(
        metadata["python_version"] == 3 and metadata["numpy_algorithm"] == "MT19937",
        "Unknown Python/NumPy RNG format",
    )
    count = integer(metadata["cuda_device_count"], "RNG device count", 0, 1)
    require(count == torch.cuda.device_count(), "RNG CPU/CUDA device consistency mismatch")
    require(
        set(tensors) == {"python_mt", "numpy_mt", "torch_rng"} | ({"cuda_rng"} if count else set()),
        "Missing or extra RNG tensor state",
    )
    for name, length in (("python_mt", 625), ("numpy_mt", 624)):
        value = tensors[name]
        require(
            value.device.type == "cpu"
            and value.dtype == torch.int64
            and tuple(value.shape) == (length,),
            "Invalid exact Mersenne Twister state",
        )
        require(bool(((value >= 0) & (value <= 2**32 - 1)).all()), "RNG values out of range")
    for name in ("torch_rng", *(("cuda_rng",) if count else ())):
        value = tensors[name]
        require(
            value.device.type == "cpu"
            and value.dtype == torch.uint8
            and value.ndim == 1
            and 1 <= value.numel() <= 65536,
            "Invalid Torch RNG state",
        )
    gaussian = metadata["python_gaussian"]
    if gaussian is not None:
        finite(gaussian, "full precision Python Gaussian cache")
    finite(metadata["numpy_gaussian"], "full precision NumPy Gaussian cache")
    integer(metadata["numpy_position"], "NumPy RNG position", 0, 624)
    integer(metadata["numpy_has_gaussian"], "NumPy Gaussian flag", 0, 1)
    random.setstate((3, tuple(tensors["python_mt"].tolist()), gaussian))
    np.random.set_state(
        (
            "MT19937",
            tensors["numpy_mt"].numpy().astype("uint32"),
            metadata["numpy_position"],
            metadata["numpy_has_gaussian"],
            metadata["numpy_gaussian"],
        )
    )
    torch.set_rng_state(tensors["torch_rng"])
    if count:
        torch.cuda.set_rng_state(tensors["cuda_rng"], 0)


class CheckpointDataStream:
    """Checkpointed native DataLoader sampling without Accelerate's one-batch read-ahead."""

    def __init__(self, prepared_loader, *, seed: int):
        import torch
        from lerobot.datasets.sampler import EpisodeAwareSampler
        from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler

        require(prepared_loader.num_workers == 0, "Exact checkpoint requires zero data workers")
        require(
            isinstance(prepared_loader.batch_sampler, BatchSampler)
            and not prepared_loader.batch_sampler.drop_last,
            "Unsupported batched/distributed or dropped data sampler",
        )
        self.dataset = prepared_loader.dataset
        size = integer(len(self.dataset), "bounded dataset frames", 1, MAX_DATASET_FRAMES)
        self.batch_size = integer(prepared_loader.batch_sampler.batch_size, "batch size", 1, 64)
        require(
            prepared_loader.batch_size in (None, self.batch_size),
            "Loader batch configuration mismatch",
        )
        sampler = prepared_loader.batch_sampler.sampler
        if isinstance(sampler, EpisodeAwareSampler):
            require(sampler.shuffle is True, "Unexpected native episode sampler")
            indices = tuple(sampler.indices)
        else:
            require(
                isinstance(sampler, (RandomSampler, SequentialSampler)),
                "Unknown native data sampler",
            )
            if isinstance(sampler, RandomSampler):
                require(
                    not sampler.replacement and sampler.num_samples == size,
                    "Replacement sampler is unsupported",
                )
            indices = tuple(range(size))
        require(bool(indices) and len(set(indices)) == len(indices), "Empty/duplicate data indices")
        for index in indices:
            integer(index, "source data index", 0, size - 1)
        self.indices, self.dataset_size = indices, size
        self.indices_sha256 = digest(canonical(indices))
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.loader_generator = torch.Generator(device="cpu").manual_seed(seed ^ 0x5A5A)
        self.epoch, self.cursor, self.batches = 0, 0, 0
        self._shuffle()
        owner = self

        class Sampler:
            def __iter__(self):
                while owner.cursor < len(owner.order):
                    index = int(owner.order[owner.cursor])
                    owner.cursor += 1
                    yield index

            def __len__(self):
                return len(owner.indices)

        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=Sampler(),
            collate_fn=prepared_loader.collate_fn,
            num_workers=0,
            drop_last=False,
            pin_memory=prepared_loader.pin_memory,
            generator=self.loader_generator,
        )
        self.device = getattr(prepared_loader, "device", None)
        self.iterator = iter(self.loader)

    def _shuffle(self) -> None:
        import torch

        indices = torch.tensor(self.indices, dtype=torch.int64)
        self.order = indices[torch.randperm(len(indices), generator=self.generator)]

    def __iter__(self):
        return self

    def __next__(self):
        from accelerate.utils import send_to_device

        if self.cursor == len(self.order):
            self.epoch += 1
            self.cursor = 0
            self._shuffle()
            self.iterator = iter(self.loader)
        batch = next(self.iterator)
        self.batches += 1
        return send_to_device(batch, self.device) if self.device is not None else batch

    def state(self) -> tuple[dict, dict]:
        return (
            {
                "schema": DATA_SCHEMA,
                "dataset_size": self.dataset_size,
                "indices_sha256": self.indices_sha256,
                "batch_size": self.batch_size,
                "epoch": self.epoch,
                "cursor": self.cursor,
                "batches": self.batches,
            },
            {
                "sampler_order": self.order.clone(),
                "sampler_rng": self.generator.get_state().clone(),
                "loader_rng": self.loader_generator.get_state().clone(),
            },
        )

    def load(self, metadata: dict, tensors: dict, *, expected_step: int) -> None:
        import torch

        keys(
            metadata,
            {
                "schema",
                "dataset_size",
                "indices_sha256",
                "batch_size",
                "epoch",
                "cursor",
                "batches",
            },
            "exact data order",
        )
        require(
            metadata["schema"] == DATA_SCHEMA
            and metadata["dataset_size"] == self.dataset_size
            and metadata["indices_sha256"] == self.indices_sha256
            and metadata["batch_size"] == self.batch_size,
            "Checkpoint sampler is bound to a different dataset/batch",
        )
        integer(metadata["epoch"], "sampler epoch")
        cursor = integer(metadata["cursor"], "consumed data cursor", 0, len(self.indices))
        require(
            cursor % self.batch_size == 0 or cursor == len(self.indices),
            "Incomplete consumed batch",
        )
        integer(metadata["batches"], "completed consumed batches", expected_step, expected_step)
        require(
            expected_step
            == metadata["epoch"] * math_batches(len(self.indices), self.batch_size)
            + math_batches(cursor, self.batch_size),
            "Sampler cursor/epoch does not equal the saved optimizer step",
        )
        require(
            set(tensors) == {"sampler_order", "sampler_rng", "loader_rng"},
            "Incomplete sampler tensors",
        )
        order = tensors["sampler_order"]
        require(
            order.dtype == torch.int64
            and order.device.type == "cpu"
            and tuple(order.shape) == (len(self.indices),),
            "Wrong sampler permutation shape",
        )
        require(
            torch.equal(
                torch.sort(order).values, torch.tensor(sorted(self.indices), dtype=torch.int64)
            ),
            "Sampler permutation does not contain exactly the approved dataset",
        )
        for key in ("sampler_rng", "loader_rng"):
            value = tensors[key]
            require(
                value.dtype == torch.uint8
                and value.device.type == "cpu"
                and value.ndim == 1
                and value.numel() <= 65536,
                "Invalid data RNG state",
            )
        self.order = order.clone()
        self.epoch, self.cursor, self.batches = metadata["epoch"], cursor, expected_step
        self.generator.set_state(tensors["sampler_rng"])
        self.loader_generator.set_state(tensors["loader_rng"])
        self.iterator = iter(self.loader)
        # With zero workers, creating an iterator consumes only its isolated base-seed draw.
        # Do not let the additional resumed iterator advance the saved future epoch stream.
        self.loader_generator.set_state(tensors["loader_rng"])


def math_batches(count: int, size: int) -> int:
    return (count + size - 1) // size


class ContinuationState:
    def __init__(self, *, seed: int, resume_root: Path | None = None, resume_step: int = 0):
        self.seed, self.resume_root, self.resume_step = seed, resume_root, resume_step
        self.stream = None

    def cycle(self, loader):
        require(self.stream is None, "Native trainer constructed more than one data stream")
        self.stream = CheckpointDataStream(loader, seed=self.seed)
        if self.resume_root is not None:
            self.restore(self.resume_root, self.resume_step)
        return self.stream

    def save(self, root: Path, step: int, rng: tuple[dict, dict]) -> None:
        from safetensors.torch import save_file

        require(
            self.stream is not None and self.stream.batches == step,
            "Completed optimizer step does not match consumed dataset batches",
        )
        data, tensors = self.stream.state()
        rng_metadata, rng_tensors = rng
        tensors.update(rng_tensors)
        state_dir = root / "training_state"
        save_file(tensors, str(state_dir / "continuation.safetensors"))
        write_json(
            state_dir / "continuation.json",
            {
                "schema": CONTINUATION_SCHEMA,
                "step": step,
                "sampler": data,
                "rng": rng_metadata,
                "precision": precision_state(),
                "num_workers": 0,
                "world_size": 1,
                "mixed_precision": "no",
            },
        )

    def restore(self, root: Path, step: int) -> None:
        from safetensors.torch import load_file

        metadata = continuation_metadata(root, step)
        require(
            metadata["schema"] == CONTINUATION_SCHEMA
            and metadata["step"] == step
            and metadata["num_workers"] == 0
            and metadata["world_size"] == 1
            and metadata["mixed_precision"] == "no"
            and canonical(metadata["precision"]) == canonical(precision_state()),
            "Full-state execution/precision/step differs; weights-only requires separate approval",
        )
        tensors = load_file(str(root / "training_state" / "continuation.safetensors"), device="cpu")
        data_names = {"sampler_order", "sampler_rng", "loader_rng"}
        require(data_names.issubset(tensors), "Missing consumed data-state tensors")
        self.stream.load(
            metadata["sampler"], {key: tensors[key] for key in data_names}, expected_step=step
        )
        restore_rng(
            (
                metadata["rng"],
                {key: value for key, value in tensors.items() if key not in data_names},
            )
        )
