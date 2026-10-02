# 개발·기여 가이드

시작 전에 [현재 상태](docs/status.md)와 [코드 구조](docs/repository-map.md)를 확인합니다.
개발 검증은 CPU/test-only이고, 실제 Azure/GPU 실행은 [운영 런북](docs/operator-runbook.md)의
별도 승인 경계를 따릅니다.

## 환경 준비

저장소 root에서 Ubuntu WSL/Linux로 실행합니다. Windows Python/Node와 섞지 않습니다.
`setup-dev.sh`는 per-worktree Linux cache를 준비하고 web 의존성을 연결합니다.
이미 준비된 환경에서는 `dev-env.sh`를 source하면 됩니다.

```bash
bash scripts/setup-dev.sh
source scripts/dev-env.sh
uv run --locked python -m contracts.validate_environment examples/inspection-cell.json --json
uv run --locked python -m scripts.check_docs
```

root API 개발 Python과 `learning/smolvla`의 pinned native Python 3.11, Isaac Python 3.12는
각각 별도 런타임입니다. interpreter/venv를 복사해 호환성을 가정하지 않습니다.
일부 app-managed Windows worktree의 `.git` pointer는 Linux Git이 해석하지 못합니다.
이 경우 source/test는 WSL, Git은 Windows에서 실행하고 `.git`나 main checkout을 수정하지 않습니다.

## 검증 명령

전체 개발 검사는 다음과 같습니다. cloud login이나 GPU 배포를 하지 않습니다.

```bash
source scripts/dev-env.sh
bash scripts/check.sh
uv run --locked python -m scripts.validate_infra
```

작은 변경은 관련 selector로 먼저 검사합니다.

```bash
uv run --locked ruff check scripts/check_docs.py tests/test_docs.py
uv run --locked ruff format --check scripts/check_docs.py tests/test_docs.py
uv run --locked pytest tests/test_docs.py -q
uv run --locked python -m scripts.check_docs
```

web 변경은 `apps/web`에서 `npm run typecheck`, `npm test`, `npm run build`를 실행합니다.
browser harness는 `npm run test:e2e`이며 fixture만 사용하고 production build에 포함하지 않습니다.
native policy/AML 인터페이스 검사는 [CI](.github/workflows/integration.yml)의 별도 pinned 환경을
따릅니다. CPU import 성공은 CUDA/renderer/학습/물리 평가 성공과 다릅니다.

## 변경 경계

- existing helper·schema·validation을 재사용하고 invalid input을 명시적으로 거부합니다.
- frozen servo/profile·joint/physics/speed/deadline 기준을 문서 정리 때문에 바꾸지 않습니다.
- source file 이동·formatter·추가 파일도 image source inventory를 바꿀 수 있습니다.
  기존 image/descriptor/certificate는 그대로 두고 새 qualification을 준비합니다.
- standalone AML Command와 API TrainingRun을 구분합니다. 외부 학습은 post-hoc import로 기록합니다.
- checkpoint full-state와 weights-only를 구분합니다. recipe/source/config/runtime 불일치는 거부합니다.
- 실제 dataset·model·checkpoint·token·signed URL·고객 자료·개인 로컬 evidence는 Git에 넣지 않습니다.
- unrelated dirty files를 덮어쓰거나 리팩토링 범위를 넓히지 않습니다.

## 문서와 PR

사용자가 처음 읽는 README는 개요·준비 상태·문서 링크만 담당합니다.
최신 사실은 [status](docs/status.md), 운영 절차는 [runbook](docs/operator-runbook.md),
구체적 계약은 각 기술 가이드, 과거 실패는 [history](docs/implementation-history.md)에 둡니다.

문서 링크 검사는 repo-local Markdown target의 파일 존재와 경로 범위를 검사합니다.
HTTP 접근성·heading anchor·코드 실행·cloud readiness는 별도 검증입니다.
문서 파일 이동 시 인덱스와 링크를 함께 갱신합니다.

PR에는 변경 목적, 실제 실행한 검사, 아직 미검증인 cloud/GPU 경계를 기록합니다.
CI가 성공해도 learned-policy release gate는 별도입니다. secrets나 비공개 artifacts를
PR body에 붙이지 않습니다. required checks를 우회하거나 historic proof를 새 commit에 재사용하지 않습니다.
