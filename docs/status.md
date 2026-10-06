# 현재 구현·검증 상태

**기준: 2026-10-03 KST. 전체 고객용 learned-policy release는 아직 차단입니다.**
이번 에셋에는 실제 장비가 없으며 시뮬레이터·학습·개발 가이드를 제공합니다.
설계 리뷰나 source test를 실제 고객 배포·로봇 품질 통과로 계산하지 않습니다.

## 무엇을 했는가

| 영역 | 실제 수행/구현 | 아직 의미하지 않는 것 |
|---|---|---|
| Web/API | React/TypeScript console, Entra FastAPI, Cosmos/private Blob, 인증·scope·승인 경계; 실제 Azure 동작 확인 | 고객별 cold deployment 완료, anonymous 실행 승인 |
| Foundry | 실제 검사 연결, 별도 proposal-only learning coach, 위험한 한도 완화 제안 거부 | agent가 motor 명령·학습 제출·release를 자동 승인 |
| Isaac Sim | 실제 렌더링·PhysX, managed Batch reference grasp/place/release/settle와 raw manifest 재검증 | learned policy의 물리 성공 |
| P0 원본 데이터 | reference controller TRAIN10001–10020: 20 episodes, 8,587 frames | 인간 시연 또는 held-out evaluation |
| P0 원본 학습 | A100에서 실제 1,000 updates, 변경된 weights, 10 full-state checkpoints, 독립 step-600 복원 | 충분한 학습량 또는 물리 품질 |
| Private 반입 | 원본 native provenance를 검증한 external-native post-hoc import | 새 API TrainingRun, customer 소유권 이전, policy release |
| P0 learned trial | finger joint 2 guard가 action 적용 전에 거부 | 완전한 scored failure/success; 결과는 incomplete/unscorable |
| P1 추가 데이터 | TRAIN11001–11007의 7 episodes, 3,015 frames; 원본 11005 실패와 승인된 별도 재시도 보존 | 추가 TRAIN20 완료; 11008–11020은 미시도, P1 모델 없음 |
| TRAIN 감사 | 원본 60 anchor: 출력 finite, 37/60 full horizons가 finger bounds 거부 | held-out 품질이나 root cause 확정 |
| Training 수정 | native temporal padding key alias를 명시적 v2 recipe로 적용; 기존 reduction·inference·guard 유지 | inference clamp 또는 물리 한도 완화 |
| Capacity | batch 1/8/16/32/64 실제 native backward, zero optimizer, digest 불변 확인 | 2,700 updates·checkpoint 전체 전송 완료 보장 |
| P0 재학습 | 아래 별도 작업은 실패. step-100/200의 각 15개 full-state 파일을 hash 검증·TTL 밖 보존 | 최종 candidate 또는 2,700 updates 완료 |
| 개발 가이드 | simulator-first 의사코드·계약·두 독립 아키텍처 리뷰 | 실행 가능한 hardware adapter, 서로 다른 모델 사용 증명 |

원본 P0 1,000 updates는 `1,000 / 8,587 ≈ 0.116` anchor passes였습니다.
scalar training loss만으로 Franka 9-joint task 품질을 판정하지 않습니다.
재학습 checkpoint의 cumulative 값은 원본 P0를 포함하므로 새 update 수와 구분합니다.

## 최근 재학습의 정확한 상태

| 항목 | 값/근거 |
|---|---|
| Azure ML job | `6cca7a7a-bd17-40f2-944c-6654f7c33d51` |
| 의도 | 원본 TRAIN20, genuine P0 weights-only 시작, batch 64, 2,700 new updates, checkpoint interval 100 |
| 결과 | `Failed`; `candidate_complete=false`, `learning_quality_verified=false` |
| 확인된 부분 결과 | step-100/200 full-state markers와 각 15개 파일의 source/destination hash 확인 |
| 보존 | 같은 private scope의 `learning/reproducibility/partial-<job-id>/checkpoints/step-000100` 및 `step-000200` |
| 실패 근거 | Low-Priority preemption 경고, 이후 native process exit 1, private `ContractError` failure marker |
| 불확실성 | 정확한 최종 update 수·직접 실패 원인 미확정. 회수 경고만으로 원인을 단정하지 않음 |
| 로그 한계 | 로컬 private Blob 403; MI reader로 failure/checkpoint는 복구. 조회한 AML log 목록과 예상 private log prefix는 비어 있음 |
| 종료 | A100 current/target nodes 0. 세 개 진단/보존 CPU execution도 종료 확인 |
| 후속 코드 | 실패 traceback 최대 64 KiB와 truncation flag 보존. **기존 실패 이미지에는 없음**; 새 image qualification 필요 |

checkpoint marker 존재만으로 모든 파일을 검증했다고 말하지 않습니다. 위 두 checkpoint는
실제 모든 파일을 별도로 읽고 복사·재조회한 경우입니다. 보존은 복원 테스트나 품질 평가와 다릅니다.
원본 실패 job/UUID/승인/deadline을 수정하거나 동일 제출을 재시도하지 않았습니다.

## 실패 진단 테스트 보강 (코드 변경, 미실행)

`learning/paused/command.py`의 `_preserve_failure`는 이미 traceback/로그 보존을 구현하고
있었으나, 기존 테스트는 두 가지 경로를 확인하지 않았습니다. 이번에 `tests/learning/`에
다음을 추가로 확인했습니다 (실행: `uv run --locked pytest tests/learning`, 994 passed /
45 skipped, Azure 호출 없음):

- `training.log`뿐 아니라 `training-context.json`도 실제 failure prefix로 발행되는지 명시적으로 확인.
- `training.log`가 고정 16 MiB budget을 넘으면 조용히 잘리지 않고 `ContractError`로 거부되는지 확인.

이 변경은 `learning/paused/command.py` 자체를 수정하지 않았으므로 AML command에 embedding되는
`CODE_FILES` snapshot sha256은 그대로입니다. 테스트 전용 변경은 qualification 재실행의 필요
조건이 아닙니다.

## 다음 제출을 위한 qualification 계획 (미실행, 승인 전 제출 금지)

과거 job `6cca7a7a-...`의 plan/job.json/snapshot은 해당 job 전용이며, 코드나 config가 바뀌면
`learning.smolvla.azure`가 새 `snapshot_sha256`/`job_sha256`/`plan_sha256`을 계산합니다. 이름만
바꿔 과거 qualification을 재사용하지 않습니다. 실제 코드를 바꾸는 경우 제출 전 아래 순서를 따릅니다.

1. `uv run --locked pytest tests/learning`로 변경된 source 전체 회귀를 확인합니다 (Azure 미접촉).
2. `uv run --locked python -m learning.checks.command_job_check --report <path>`와
   `uv run --locked python -m learning.checks.embedded_source_check --report <path>`를 실행해
   SDK root 정규화·zero code/data-asset resolution·cold expiry 거부를 offline으로 재확인합니다.
3. 새 plan을 생성해 `plan.json`의 `snapshot_sha256`이 실제 변경된 `CODE_FILES` 내용과 일치하는지
   확인하고, 과거 `6cca7a7a-...` plan/job 디렉터리를 덮어쓰지 않습니다 (새 output 경로 필수).
4. 새 plan/snapshot hash, 변경된 파일 목록, 위 1–2 실행 결과를 인수인계에 기록한 뒤에만
   승인자가 명시적으로 실제 제출(=유료 GPU 시작)을 승인합니다. 이 세션은 그 승인을 수행하지 않았고
   Azure에 어떤 쓰기도 하지 않았습니다.

## 남은 작업 순서

| 우선순위 | 작업 | 완료 기준 |
|---|---|---|
| 1 | 실패 진단과 prospective 복구 계획 | 가능한 원본 오류 증거 확보; 원인 미확정이면 그 한계를 명시. 새 source/image·입력·실행 identity 검증 |
| 2 | 새 학습 또는 정확한 full-state resume qualification | checkpoint의 recipe/source/runtime/config/horizon bindings 통과. old checkpoint를 다른 recipe로 강제 resume하지 않음 |
| 3 | 실제 학습 완료 및 checkpoint custody | 실제 update 수·changed weights·완전한 manifest·독립 복원·원본 log·cleanup |
| 4 | 새로운 import/serving context | 원래 1,000-step/3,600-second context를 덮어쓰지 않는 승인; 새 verifier/runtime descriptor qualification |
| 5 | 독립 물리 평가 | 고정된 disjoint 평가 protocol, 모든 trial 원본 증거, 기준 성공률·개선·zero-safety 조건. incomplete는 성공으로 계산하지 않음 |
| 6 | 고객 scope cold deployment/워크샵 | 고객 identity·artifact·quota·image·scene 준비와 실제 리허설. 다른 owner artifact 접근 우회 없음 |
| 선택 | 추가 P1 cohort | 11008–11020을 별도 승인 수집하여 전체 데이터 검증; 현재 P0 복구와 혼동하지 않음 |
| 미래 | ROS/MQTT/MCP 또는 실물 | 선택 adapter와 별도 계약·qualification. 이번 에셋의 필수 완료 조건 아님 |

held-out 데이터로 학습/tuning하지 않고, joint/speed/physics 한도나 frozen 39-file bundle을
수정하지 않습니다. Paused 결과는 **NON_REALTIME_SIMULATION**이며 real-time 승인으로 승격하지 않습니다.

## 상태 갱신과 증거 위치

Git은 코드·계약·가이드·비민감 요약을 보관합니다. 실제 데이터·weights·checkpoint·승인·실행
receipt는 private scope에 보관하며 Git에 복사하지 않습니다. tenant/owner/resource ID는 고객
배포에서 새로 결정합니다. 개발자의 로컬 세션 폴더를 고객 런북 prerequisite로 만들지 않습니다.

새 결과가 생기면 이 문서와 [고객 runbook](customer-workshop.md)의 준비 상태를 함께 갱신하고,
원래 실패 결과는 [구현 이력](implementation-history.md) 및 private evidence에 남깁니다.
[운영 절차](operator-runbook.md)는 상태를 확인하는 방법이며 실행 완료 증명이 아닙니다.
