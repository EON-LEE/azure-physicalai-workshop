from dataclasses import dataclass, field


@dataclass
class Problem(Exception):
    status: int
    code: str
    message: str
    details: list[dict] = field(default_factory=list)
    retryable: bool = False

    def __str__(self) -> str:
        return self.message


def unavailable(component: str) -> Problem:
    return Problem(
        503,
        "dependency_unavailable",
        f"{component} is unavailable; no fallback was used.",
        retryable=True,
    )
