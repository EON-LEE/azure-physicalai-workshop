# 문서 인덱스

[저장소 README](../README.md) → [현재 상태](status.md) → 목적별 가이드 순서로 읽습니다.
**구현됨, 실제 실행됨, 물리 품질 검증됨, 고객 release됨은 서로 다른 상태입니다.**
날짜가 있는 과거 기록보다 `status.md`의 최신 결과를 우선하되 원본 증거를 수정하지 않습니다.

## 시작 및 운영

| 문서 | 다루는 내용 |
|---|---|
| [현재 상태](status.md) | 실제 수행 결과, 남은 차단 조건, 다음 작업 순서 |
| [고객 워크샵](customer-workshop.md) | 장비 없는 시나리오, 조건부 참가자 흐름, 코드 의사코드, 선택 확장 |
| [운영 런북](operator-runbook.md) | preflight, 제출, 관측, 실패, checkpoint, cleanup, 인수인계 |
| [개발 가이드](../CONTRIBUTING.md) | WSL 환경, 검증 명령, PR와 문서 갱신 규칙 |
| [코드 구조](repository-map.md) | 디렉터리 책임, 실제 entrypoint, 안전한 변경 경계 |
| [구현 이력](implementation-history.md) | 이전 긴 README의 날짜별 기술 이력; 현재 상태 판정에는 사용하지 않음 |

## 배포·사용자 인터페이스

| 문서 | 다루는 내용 |
|---|---|
| [Azure deployment](azure-deployment.md) | staged deployment, identity/private network, legacy VM와 managed 경로 구분 |
| [GPU cost and capacity](gpu-cost-options.md) | quota·Spot·renderer 의존성, 비용 추정의 한계 |
| [Customer environments](customer-environments.md) | JSON/Python scene 확장, 지원 template와 configuration-only 예제 |
| [HTTP API](http-api.md) | private API, 승인·ACK·실제 결과의 구분 |
| [Public demo API](public-demo-api.md) | synthetic presentation 공개 범위와 읽기 전용 경계 |
| [Learning API](learning-api.md) | Teaching Studio, owner scope, 학습/반입/release 계약 |

## 시뮬레이션·학습

| 문서 | 다루는 내용 |
|---|---|
| [Runtime control](runtime-control.md) | 제어 guard·상태·이미지 출처·안전 경계 |
| [Paused simulation](paused-simulation.md) | non-real-time 계약, v1/v2, physics time와 wall time |
| [Managed simulation](managed-simulation.md) | private Azure Batch pool/task와 reference 실행 |
| [Policy learning](policy-learning.md) | 모델 선택·라이선스·데이터·AML·checkpoint |
| [TRAIN-only audit](smolvla-train-audit.md) | 원본 TRAIN 예측 감사; held-out 물리 평가의 대체가 아님 |
| [Training padding](smolvla-training-padding.md) | opt-in native mask alias, provenance, batch capacity 검증 |
| [One learned evaluation](managed-learned-evaluation.md) | mixed Python image와 단일 learned physical trial |
| [Paired evaluation](managed-paired-evaluation.md) | disjoint seed·짝지은 평가·완전한 raw evidence |

## 검증

| 문서 | 다루는 내용 |
|---|---|
| [Automated testing](automated-testing.md) | CPU/UI/IaC와 실제 cloud/GPU gate 구분 |
| [Live acceptance](live-acceptance.md) | 승인된 실제 endpoint 검사와 release aggregation |
| [Test catalog](../tests/cases.json) | 구현된 case와 계획된 case의 machine-readable 구분 |

## 문서 관리 규칙

- `status.md`는 최신 운영 사실과 차단 조건의 단일 요약입니다. 날짜·job ID·증거 수준을 함께 갱신합니다.
- 런북은 재현 가능한 절차, 개별 기술 가이드는 계약, 이력은 과거 사건을 담당합니다.
- 실행 명령과 의사코드를 구분하고, placeholder는 실행 가능한 값처럼 표시하지 않습니다.
- 자격 증명·token·signed URL·고객 데이터·weights·세션 임시 파일을 커밋하지 않습니다.
- 파일 이동 시 링크를 갱신하고 `python -m scripts.check_docs`를 실행합니다.
- source/image/model fingerprint가 바뀌면 재qualification 필요성을 문서화합니다.
