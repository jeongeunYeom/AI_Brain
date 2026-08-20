# Petroleum RAG Agent Workspace

Agent가 읽고 생성하고 수정할 연구 파일을 이 폴더 안에 넣으세요.

기본 설정은 `.env`의 다음 값으로 변경할 수 있습니다.

```env
AGENT_WORKSPACE_DIR=./workspace
```

Agent는 이 폴더 밖의 경로, `.env`, 인증키·토큰 파일, 파일 삭제, 임의 셸 명령, 인터넷 접근을 허용하지 않습니다.
생성된 작업 기록은 `data/agent_runs/`, 수정 전 백업은 `data/agent_backups/`에 저장됩니다.
