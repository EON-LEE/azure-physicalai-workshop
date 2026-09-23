"""Short immutable-image entry point; new profiling runs require an explicit UTC start deadline."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from learning.checks.smolvla_vendor_profile import (
    run_profile,
    self_test_spawn,
    self_test_timeout,
    validate_specification,
)
from learning.common import canonical, digest, file_digest, read_json, require, sha256


def materialize_specification(
    path: Path,
    *,
    template_sha256: str,
    image: str,
    configuration_sha256: str,
) -> dict:
    require(
        file_digest(path) == sha256(template_sha256), "Immutable profile template checksum mismatch"
    )
    template = read_json(path, max_bytes=32768)
    validate_specification(template)
    result = copy.deepcopy(template)
    # An image cannot embed its own digest; the reviewed job pins this sole metadata overlay.
    result["vendor_specification"]["image"] = image
    validate_specification(result)
    require(
        digest(canonical(result)) == sha256(configuration_sha256),
        "Materialized profile configuration differs from the reviewed job",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration-file", type=Path)
    parser.add_argument("--template-sha256")
    parser.add_argument("--image")
    parser.add_argument("--configuration-sha256")
    parser.add_argument("--self-test-spawn", action="store_true")
    parser.add_argument("--self-test-timeout", action="store_true")
    args = parser.parse_args()
    if args.self_test_spawn or args.self_test_timeout:
        require(
            not any(
                (
                    args.configuration_file,
                    args.template_sha256,
                    args.image,
                    args.configuration_sha256,
                )
            ),
            "Spawn self-tests never execute a model specification",
        )
        report = self_test_timeout() if args.self_test_timeout else self_test_spawn()
        print(json.dumps(report, indent=2))
        return
    require(
        all((args.configuration_file, args.template_sha256, args.image, args.configuration_sha256)),
        "Explicit configuration file/hash and immutable runtime image are required",
    )
    run_profile(
        materialize_specification(
            args.configuration_file,
            template_sha256=args.template_sha256,
            image=args.image,
            configuration_sha256=args.configuration_sha256,
        )
    )


if __name__ == "__main__":
    main()
