# AI_Brain Research Agent v1

## 제공 기능

- 작업공간 내부 폴더 탐색
- `.txt`, `.md`, `.py`, `.json`, `.csv`, `.yaml`, `.yml`, `.log` 읽기
- 기존 파일을 덮어쓰지 않는 새 파일 생성
- 정확히 한 부분만 바꾸는 승인 기반 파일 수정
- 수정 전 `data/agent_backups/<task-id>/` 자동 백업
- 제한된 Python 데이터 분석 실행
- 계획, 도구, 파일, 코드, 결과, 오류를 `data/agent_runs/<task-id>.json`에 기록

기존 RAG API와 화면은 유지되며 `/agent` 화면에서 Agent 모드로 전환할 수 있습니다.

## 설정

`.env.example`을 참고하여 작업공간과 실행 제한을 설정합니다.

```env
AGENT_WORKSPACE_DIR=./workspace
AGENT_PYTHON_TIMEOUT_SECONDS=30
AGENT_MAX_FILE_BYTES=5242880
```

## API

- `POST /api/agent/plan`
- `POST /api/agent/tasks/{task_id}/execute`
- `GET /api/agent/tasks/{task_id}`
- `POST /api/agent/tasks/{task_id}/cancel`
- `GET /api/agent/workspace?path=.`

## 안전 제한

Python은 별도 프로세스로 실행되며 실행 시간, 코드 길이와 출력 길이를 제한합니다. 사용자 코드의 import는 데이터 분석용 허용 목록으로 제한하고, 네트워크 소켓과 작업공간 밖의 파일 열기를 런타임에서도 차단합니다. 이 방식은 운영체제 수준 컨테이너 격리는 아니므로 신뢰할 수 없는 다중 사용자 서버에 공개하기 전에는 컨테이너 또는 별도 OS 사용자 격리를 추가해야 합니다.

1차 버전에서는 파일 삭제, 이동, 임의 셸 명령, 관리자 권한, 패키지 설치, 인터넷 검색, Git 자동 작업을 제공하지 않습니다.
