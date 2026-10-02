# 저장소 구조와 변경 경계

이번 정리는 문서 책임·탐색·entrypoint를 리팩토링합니다. runtime package를 임의로
이동하지 않습니다. 학습 이미지와 simulator descriptor가 전체 source inventory를
attest하므로 경로만 바꿔도 historical proof를 재사용할 수 없기 때문입니다.

## 디렉터리 책임

| 경로 | 책임 | 대표 entrypoint/계약 |
|---|---|---|
| `apps/web` | React console, public viewer, MSAL operator, production decoder | `package.json` scripts; fixture browser harness는 tests 전용 |
| `apps/api` | Entra API, scope/revision/approval/run/result | `main.py`, `models.py`, `learning_models.py` |
| `apps/learning_worker` | private MI job/registry/import 경계 | `external_training.py`; externally trained import와 API run 구분 |
| `agents` | inspection와 proposal-only coach | Foundry adapter·versioned instructions |
| `contracts` | 고객 JSON schema/offline validator | `python -m contracts.validate_environment` |
| `simulation` | Isaac scene/guard/bridge, Batch orchestration | `batch.py`, `batch_learned.py`, `command_policy_deployment.py` |
| `learning` root | shared contract, conversion, deadlines, real-time 경로 | `contract.py`, `inference.py`; paused 경로와 혼용 금지 |
| `learning/paused` | 명시적 non-real-time capture/command/IPC/evaluation | `command.py`, `ipc.py`, `inference.py`, `train_audit.py`, `train_capacity.py` |
| `learning/smolvla` | native model/trainer/checkpoint/AML image | `azure.py`, `checkpoint_runner.py`, `image_bootstrap.py`, 별도 lock |
| `learning/gr00t` | 기존 별도 모델 검토·계약 코드 | SmolVLA fallback으로 자동 전환하는 production 경로가 아님 |
| `infra` | staged Azure IaC | foundation, learning, simulation Batch와 access |
| `scripts` | 개발 검증·배포·live acceptance | `check.sh`, `deploy.py`, `release_gate.py`, `check_docs.py` |
| `examples` | 검토된 reference/custom 설정과 configuration-only 예제 | `inspection-cell.json`, `compact-cell.json`, `customer-cell.json` |
| `tests` | CPU/HTTP/browser 계약·live catalog | `cases.json`, `learning/`, web tests는 `apps/web` |
| `docs` | 최신 상태·런북·기술 계약·이력 | [문서 인덱스](README.md) |
| `test-results` | 로컬 test-only 산출물 | Git ignored; 실제 customer/GPU proof 대체가 아님 |

## 네 가지 execution 경로를 구분

| 경로 | 용도 | 주의 |
|---|---|---|
| Reference inspection bridge | Foundry 검사 뒤 기준 제어기 grasp/sort | 성공해도 learned skill이 아님; legacy private VM 문서 포함 |
| Managed Batch reference | 실제 Isaac 시연 수집/원본 physical episode | reference dataset; seed·manifest·scope·deadline 고정 |
| Standalone AML Command | SmolVLA 실제 학습·checkpoint | CPU schema만으로 submitted/completed라고 주장하지 않음 |
| Managed learned evaluation | pinned learned model의 단일 trial 또는 paired 평가 | source/runtime/image descriptor; guard 거부/incomplete 결과 보존 |

## 변경 시 확인 순서

1. 소유 package와 schema를 확인하고 기존 helper·테스트를 찾습니다.
2. 영향을 받는 API/web decoder, data/model/provenance, 이미지 source inventory를 확인합니다.
3. 가장 작은 검사부터 실행하고 CI로 전체 accumulated branch를 검증합니다.
4. README/status/runbook에 구현/실제 실행/미검증을 구분해서 반영합니다.
5. 새 GPU/cloud 실행은 별도 source/image qualification과 승인된 deadline으로 수행합니다.

`simulation`의 frozen control 39-file bundle과 numeric limit은 이번 문서 리팩토링으로
변경하지 않습니다. 새 failure diagnostics는 source에만 구현됐으며 기존 실패 job의
고정 이미지에 retroactive 적용하지 않습니다.

## 향후 리팩토링 기준

source 이동을 하려면 import·Docker COPY·runtime source inventory·서명/manifest·테스트
경로를 함께 변경하고 새 proof를 만들어야 합니다. 먼저 protocol별 책임을 명시적으로
유지하고, 실물 adapter는 별도 계약으로 설계합니다. 기존 `physics_step`이나 process-local
monotonic deadline을 실물 driver에 그대로 전달하는 "공통화"는 하지 않습니다.
