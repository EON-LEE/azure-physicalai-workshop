# 운영 런북: 준비 → 실행 → 증거 → 종료

**대상:** 실제 장비 없는 Azure Isaac Sim/SmolVLA 에셋의 승인된 운영자.
**현재 결과:** [status](status.md). 이 문서는 운영 방법이지 새 유료 실행 승인이나
고객 learned-policy release가 아닙니다.

## 1. 운영 모드 결정

| 목적 | 사용 경로 | 완료를 판정하는 증거 |
|---|---|---|
| 고객 설정 작성 | JSON validator, Environment Studio draft | configuration-only validation; simulator 준비 완료 아님 |
| 기준 시연 | managed Batch reference | 원본 frame·physics·grasp/place/release/settle와 complete manifest |
| 정책 학습 | 승인된 private worker/standalone AML Command | 실제 optimizer count, changed weights, full checkpoints, native completion |
| learned 평가 | 별도 Batch learned/paired spec | 같은 model/task/profile·disjoint seeds·완전한 물리 결과 |
| 공개 시연 | 승인된 synthetic presentation | same epoch 원본 image/Foundry/physical outcome; 없으면 unavailable |

현재 paused path는 **NON_REALTIME_SIMULATION**입니다. real-time 100/80 ms gate 실패를
paused 성공으로 덮지 않습니다. 이번 에셋에는 real device 또는 edge broker가 필요 없습니다.

## 2. Preflight checklist

작업 전에 다음을 기록합니다. placeholder 값을 그대로 제출하지 않습니다.

| 범주 | 필수 확인 |
|---|---|
| Identity | 명시적 subscription/tenant/owner, 실제 실행 MI와 최소 권한, 고객 scope |
| Network | 기존 private runner의 Blob/Batch/AML 접근. 로컬 403은 권한·network를 확인; public access/auth bypass로 해결 금지 |
| Inputs | 원본 dataset manifest·conversion·model·backbone hash, TRAIN/held-out 분리 |
| Runtime | digest-pinned image, actual embedded source proof, Python/LeRobot/Torch/driver qualification |
| Controls | task/profile/criteria·frozen control hash·guard·scene/license 자산 |
| Budget/lifetime | 이번 작업의 명시적 승인, 고정 absolute deadline, watchdog, CPU/GPU cleanup 계획 |
| Quiescence | 이전 작업 terminal/owned GPU node 0, duplicate claim/submission 여부 |
| Capacity | quota뿐 아니라 regional SKU·graphics/CUDA·batch memory·disk·checkpoint 전송 qualification |

campaign 비용 상한 해제가 기존 작업 deadline·권한·안전 기준을 없애지 않습니다.
새 image나 plan을 만들면 source/config hash가 바뀝니다. 과거 qualification을 이름만 바꿔 사용하지 않습니다.

## 3. 계획과 제출을 구분

개발 검증은 WSL repository root에서 실행하며 Azure를 변경하지 않습니다.

```bash
source scripts/dev-env.sh
uv run --locked python -m contracts.validate_environment examples/inspection-cell.json --json
uv run --locked python -m scripts.check_docs
uv run --locked python -m scripts.deploy --help
uv run --locked python -m simulation.batch --help
uv run --locked python -m simulation.batch_learned --help
```

`scripts.deploy CONFIG`는 지원 stage의 preview이고 `--apply`는 billable writes입니다.
이 entrypoint만 실행했다고 전체 Batch/AML workshop이 provisioned된 것은 아닙니다.
실제 configuration/spec 준비는 [배포](azure-deployment.md), [Batch](managed-simulation.md),
[learned trial](managed-learned-evaluation.md), [학습](policy-learning.md)의 계약을 따릅니다.

학습 plan 생성기 `learning.smolvla.azure`는 config·plan-dir·job-name을 받지만
실제 cloud 제출은 별도 승인된 worker/controller 경로입니다. 로컬 UUID를 다시 생성하거나
인터넷 복구 후 같은 plan을 재제출하지 않습니다. 기존 job ID를 먼저 조회합니다.
개발자의 세션 임시 Python 파일을 고객 배포 도구로 복사하지 않습니다.

## 4. 관측: 상태와 진척은 별도

다음 명령은 운영자가 이미 선택한 계정에서 Azure CLI + ML extension으로 실행하는
**조회 예제**입니다. PowerShell 또는 WSL의 자기 CLI 환경을 사용하고 출력 token을 공유하지 않습니다.
환경변수는 해당 작업의 실제 승인 값으로 설정합니다.

```powershell
az ml job show --name $env:TRAINING_JOB --workspace-name $env:AML_WORKSPACE --resource-group $env:RESOURCE_GROUP --subscription $env:AZURE_SUBSCRIPTION --query status --output tsv
az ml job stream --name $env:TRAINING_JOB --workspace-name $env:AML_WORKSPACE --resource-group $env:RESOURCE_GROUP --subscription $env:AZURE_SUBSCRIPTION
```

| 관측 | 해석 |
|---|---|
| Queued / Starting | optimizer 실행 증거가 아님 |
| Running | 프로세스 진행 상태; 실제 native log/update/checkpoint 확인 필요 |
| preemption | interrupted attempt 가능. 같은 physical episode를 몰래 재실행할 권한이 아님 |
| Completed | native receipt·files/hash·model·quality를 별도로 검증 |
| Failed / Canceled | 실패 그대로 기록; 최대 update 수를 요청값으로 채우지 않음 |
| read error / network disconnect | status UNKNOWN; terminal로 추정하거나 새 job을 제출하지 않음 |

watchdog와 observer를 구분합니다. observer shell exit 0은 관측 성공일 뿐 job 성공이 아닙니다.
cloud resource ID·job ID·image·spec/source hash·관측 시각·원본 오류를 보존합니다.

## 5. 실패·로그 복구

1. 새 GPU 제출을 멈추고 원본 terminal status와 watchdog receipt를 확인합니다.
2. native `user_logs/std_log.txt`, command failure marker, checkpoint inventory를 조회합니다.
3. 로컬 storage가 403이면 기존 승인된 private MI runner로 읽습니다. account key·public endpoint·권한 우회는 금지합니다.
4. failure type, 실제 error/traceback, publication/cleanup 오류를 구분합니다. 오류 type만 있으면 직접 root cause는 미확정입니다.
5. source에 수정이 필요하면 테스트하고 새 immutable image를 qualification합니다. historic receipts는 덮어쓰지 않습니다.

후속 source의 `_preserve_failure`는 최대 64 KiB `traceback.txt`와 truncation flag를
private failure artifact에 발행합니다. 학습 log 생성 전 오류도 보존하며 기존 실패 이미지를
소급 수정하지 않습니다. traceback/log는 private 자료이고 secrets 포함 가능성이 있어 Git에 넣지 않습니다.

최근 job의 preemption 경고와 `ContractError`는 확인됐지만 직접 원인은 아직 확정되지 않았습니다.
빈 log 조회 결과를 "오류 없음"으로 해석하지 않습니다.

## 6. Checkpoint custody와 resume

| 확인 | 절차/거부 기준 |
|---|---|
| 완전성 | marker-last checkpoint manifest와 모든 파일 존재·size·SHA·ETag 검증. 더 최신 partial folder 무시 |
| 보존 | output TTL 밖 같은 승인 scope의 reproducibility prefix에 보존; 모든 destination hash 재조회 |
| Provenance | 원본 job/source/runtime/config/recipe/task/profile/scope/horizon binding 보존 |
| Full-state | model·optimizer·scheduler·RNG·data position 복원. exact binding 불일치 거부 |
| Weights-only | fresh optimizer/scheduler/data state로 새 실행; full-state continuation이라고 설명 금지 |
| 품질 | checkpoint 저장/복원 성공은 물리 성공률 아님 |

새 job ID/deadline을 쓰려면 해당 resume contract가 이를 허용하는지 먼저 검증해야 합니다.
경로를 바꿔 binding이 바뀌는 문제를 hash 재작성으로 해결하지 않습니다.
새 모델은 기존 budget/source certificate를 바꾸지 않는 prospective import context가 필요합니다.

## 7. 평가와 고객 handoff

실제 learned trial은 pinned model·task·profile·image·spec로 별도 실행합니다.
기준 제어기 성공률, training loss, CPU fixtures, 감사 anchor를 held-out 결과로 대체하지 않습니다.
guard rejection/timeout/unknown/incomplete를 보존하고 clip/clamp로 성공을 만들지 않습니다.
고정 평가 기준은 [paired evaluation](managed-paired-evaluation.md)을 따릅니다.

고객에게 제공할 때는 고객 scope의 모델·데이터·권한·quota·image·scene·실행 리허설이 필요합니다.
동일 tenant의 다른 사용자도 같은 owner가 아닙니다. 지금 weights를 복사하는 것은 portable release가 아닙니다.
지금 진행 가능한 것은 구현 설명·reference 경로와 제한이 표시된 시연입니다.
learned 실행 단계는 [고객 문서](customer-workshop.md)의 준비 gate가 통과해야 합니다.

## 8. 종료와 인수인계

고정 deadline까지 실행을 종료하고, 이번 작업에 속한 resource ID만 대상으로 정리합니다.
wildcard resource deletion·subscription 전체 역할 변경·unrelated host 종료는 하지 않습니다.

| 인수인계 필드 | 기록 |
|---|---|
| 작업 | original ID, terminal status, 실제 update 수 또는 UNKNOWN |
| 산출물 | model/checkpoint/dataset manifest hash, custody prefix, 원본 실패 |
| 비용 | 실제 조회 근거 또는 UNKNOWN; cleanup을 비용 0의 증거로 쓰지 않음 |
| Cleanup | A100 current/target nodes 0, Batch pool/task 종료·node 0, CPU controller/watchdog terminal |
| 다음 작업 | 새 승인·qualification·import/평가 조건과 남은 blocker |
| 문서 | `status.md`와 customer readiness 갱신; 과거 실패 보존 |

[현재 상태](status.md)의 GPU 0은 마지막 관측 결과이지 영구적인 idle 보장이나 비용 청구서가 아닙니다.

## 9. 서버 이전: Git 밖의 증거 백업과 복원

Git clone은 코드만 복원합니다. 기존 실행의 승인·specification·image/source proof,
원본 모델 manifest, 실패 진단·checkpoint custody receipt, 고정 TRAIN/평가 설정과
실행 영상은 별도 private 백업이 필요합니다. `.venv`·`node_modules`·패키지 캐시는 제외합니다.
로그에 민감 정보가 있을 수 있으므로 Git, public Blob, SAS 공유 링크에 올리지 않습니다.

**2026-10-04 이전 작업 상태:** 로컬 증거 백업을 준비 중이며 Blob 업로드 완료는 아직
확인하지 않았습니다. Azure CLI의 계정 선택만으로 업로드를 증명하지 않습니다.
현재 PC의 기존 private Blob 접근은 차단됐고, 이후 로그인 서버 연결도 시간 초과됐습니다.
아래 명령은 네트워크·로그인이 준비된 운영 환경에서 수행할 절차입니다.
원본 파일·프로젝트는 원격 검증과 새 서버 접근 확인 전 삭제하지 않습니다.

### 백업 내용과 private 인수인계

| 항목 | 내용 |
|---|---|
| `physicalai-evidence.zip` | 세션 `files`, checkpoint 요약, 저장소 `test-results`의 일반 파일 |
| `inventory.json` | 파일별 상대 경로·byte 수·SHA-256, source Git commit, 제외한 symlink 경로·target |
| `receipt.json` | ZIP/외부 inventory hash, 파일 수, 로컬 검증 결과 |
| 별도 private 인수인계 | subscription·tenant·storage account·container·정확한 Blob prefix와 ZIP hash |

symlink는 외부 경로를 따라 읽지 않고 inventory에 기록합니다. 복원된 일반 파일을
우선 사용하며 native test fixture의 `last` symlink는 필요한 환경에서만 다시 만듭니다.
이 ZIP은 실제 cloud weights·전체 dataset의 백업이 아닙니다. 해당 private artifact의
현재 존재·hash·보존 기간은 custody 경로에서 따로 확인해야 합니다.
과거 plan/controller는 기록 자료이며 새 서버에서 그대로 재실행할 승인이 아닙니다.

### 업로드

같은 계정이라도 private endpoint에 도달하는 DNS·VPN 또는 기존 VNet 실행 경로와
Blob data-plane 권한이 필요합니다. `az account show` 성공은 이 조건을 증명하지 않습니다.
403을 public network 활성화·account key·역할 우회로 해결하지 않습니다.

다음 Bash 명령의 변수는 **private 인수인계의 실제 값**으로 설정합니다.
`BACKUP_DIR`에는 위 세 파일만 두고, `PREFIX`에는 기존 artifact와 겹치지 않는 새
owner-scoped migration prefix를 사용합니다.

```bash
az login --tenant "$TENANT"
az account set --subscription "$SUBSCRIPTION"
az storage blob upload-batch \
  --account-name "$STORAGE_ACCOUNT" --auth-mode login \
  --destination "$CONTAINER" --destination-path "$PREFIX" \
  --source "$BACKUP_DIR" --overwrite false
```

private MI runner를 사용하면 이미 승인된 identity로 업로드합니다. 로컬 `az login`
캐시를 runner로 복사하지 않습니다. 원본 프로젝트 전체를 image나 ARM 로그로
전송하는 대신 승인된 private 파일 전송 경로를 사용합니다.
재시도 전에 동일 prefix의 파일을 조회하고 hash를 확인합니다.
단순 list/HEAD/metadata hash 확인은 전체 byte 검증을 대신하지 않습니다.

### 새 서버 다운로드·검증

먼저 Git 저장소의 `main`을 clone하고 private 인수인계의 계정·경로를 확인합니다.
새 서버에서 아래 명령을 실행합니다. `RESTORE_DIR`는 새 빈 디렉터리여야 합니다.

```bash
az login --tenant "$TENANT"
az account set --subscription "$SUBSCRIPTION"
mkdir "$RESTORE_DIR"
az storage blob download-batch \
  --account-name "$STORAGE_ACCOUNT" --auth-mode login \
  --source "$CONTAINER" --destination "$RESTORE_DIR" \
  --pattern "$PREFIX/*"
```

download-batch는 Blob prefix의 디렉터리 구조를 유지합니다. 다운로드한 ZIP의
SHA-256을 **별도 전달받은 검증된 원본 hash**와 비교합니다.
ZIP과 같이 다운로드한 receipt만 신뢰 기준으로 사용하지 않습니다.
PowerShell에서는 `Get-FileHash -Algorithm SHA256 -LiteralPath $ArchivePath`,
Linux에서는 `sha256sum "$ARCHIVE_PATH"`를 사용합니다.

hash 일치 후 ZIP을 빈 디렉터리에 풀고 `inventory.json`의 모든 일반 파일에 대해
경로·byte 수·SHA-256을 확인합니다. 기존 checkout이나 native runtime을 덮어쓰지 않습니다.
로컬 upload 결과와 별개로 Blob에서 다시 다운로드한 ZIP 전체 hash를 확인하고,
인수 서버에서도 접근·복원을 확인해야 migration 완료입니다.
업로드·검증 시각과 정확한 prefix는 private 인수인계에 기록하며, 완료 전에는
`cloud_upload_verified=false`를 유지합니다.
