# AI_Brain · Petroleum Engineering Research Agent

석유공학 문서에 근거해 답변하는 RAG와 연구 파일을 직접 분석하는 Agent를 결합한 로컬 우선 웹 애플리케이션입니다.

AI_Brain의 목표는 단순한 질의응답을 넘어 PDF와 연구 데이터를 검색·분석하고, 보고서와 그래프를 안전하게 생성하는 **실행형 석유공학 연구지원 AI**를 만드는 것입니다.

## 핵심 목표

- 석유공학 PDF와 교재를 기반으로 출처가 있는 답변 생성
- CSV와 연구 결과를 분석하여 통계·보고서·그래프 생성
- 사용자가 작업 계획과 실행 코드를 확인한 뒤 승인
- 로컬 Ollama 모델을 사용하여 외부 API 비용 최소화
- 모든 파일 작업을 제한된 workspace 안에서 수행하고 기록
- MRST/CO₂ 저장 결과 분석과 석유공학 전용 계산 도구로 확장

## 현재 제공 기능

### 1. 문서 기반 RAG

- PDF 업로드, 텍스트 추출 및 청크 분할
- BGE-M3 임베딩과 ChromaDB 영구 저장
- 질문 유형에 따른 검색 전략 선택
- 검색 문서에 근거한 답변과 문서·페이지 출처 표시
- PDF 그림 추출, Vision 모델 분석 및 Figure Note 저장
- 관련 그림과 Plotly 그래프 표시
- Figure Review 및 성능 평가 화면

근거가 부족한 경우 모델이 임의로 답하지 않고 다음과 같이 응답하도록 구성합니다.

```text
제공된 문서 근거로는 확인할 수 없습니다.
```

### 2. 연구지원 Agent

- workspace 폴더 탐색과 파일 읽기
- CSV 열, 행 수, 표본 데이터와 평균·중앙값·표준편차·Pearson 상관계수 확인
- TXT 분석 보고서 생성
- PNG 산점도·선 그래프·막대그래프·히스토그램 생성
- 그래프 종류와 X축·Y축 열 직접 선택
- 한국어 요청에서 CSV 열 이름 인식
- 여러 CSV의 공통 숫자 열 탐지와 파일별 최소·최대·평균 비교
- 다중 CSV 비교 결과표와 PNG 그래프 동시 생성
- MRST/CO₂ Storage 전용 분석 모드
- `srco2`, 시간, trapped/free 비율 또는 양 열 자동 인식
- 비율 정규화와 trapped/free 양 기반 포획 비율 계산
- 조건·시간별 변화 분석 및 요약 CSV·PNG·Markdown 동시 생성
- 계산 기준, 입력 단위, Pearson 상관과 비인과성 주의사항 기록
- Agent에서 Text/Figure RAG 지식베이스 읽기 전용 검색
- 관련 문헌의 문서명·페이지·근거 문장과 Figure 경로 반환
- 새 파일 생성과 기존 파일 부분 수정
- 수정 전 원본 자동 백업
- 제한된 Python 데이터 분석 실행
- 작업 계획, 사용할 도구와 실행 코드를 승인 전에 표시
- 작업 취소와 진행 상태 확인
- Python 종료코드·timeout 및 요청 결과물 자동 검증
- PNG 무결성, CSV 데이터, 텍스트 내용 확인 후 완료 처리

Agent 실행 흐름:

```text
사용자 요청
→ 작업 계획 생성
→ 대상 파일·도구·실행 코드 확인
→ 사용자 승인
→ 안전성 검사 및 실행
→ 결과 검증
→ 작업 기록 저장
```

MRST/CO₂ 분석에서는 Agent 화면의 **MRST / CO₂ Storage 전용 분석**을
선택합니다. 비율 값이 `0~1`이면 백분율로 환산하며, 비율 열 없이
trapped/free 양 열만 있으면 두 값의 합을 분모로 비율을 계산합니다.
원본 CSV는 수정하지 않고 기본적으로 다음 결과를 생성합니다.

```text
results/mrst_co2_analysis.csv  # 행별 정규화 결과와 계산 기준
results/mrst_co2_analysis.png  # srCO₂ 또는 시간별 trapped/free 변화
results/mrst_co2_analysis.md   # 통계·상관·열 매핑·가정 보고서
```

### 3. 결과 미리보기와 다운로드

- PNG 그래프 미리보기 및 다운로드
- TXT·Markdown 본문 미리보기 및 다운로드
- CSV 표 미리보기 및 다운로드
- 생성 결과를 Agent 화면에서 바로 확인

### 4. 작업 기록

- 하나의 Agent 대화에 여러 작업 누적
- 새 대화 생성, 과거 대화 목록 및 대화 전체 복원
- 대화별 작업 기록 연결
- 최근 Agent 작업 50개를 최신순으로 표시
- `completed`, `failed`, `canceled` 상태 확인
- 요청 내용, 생성 시각과 실행 시간 표시
- 사용 도구와 생성·수정 결과 확인
- 작업 상세 기록 다시 열기
- 손상된 기록 파일은 전체 화면을 중단하지 않고 제외

작업 기록은 다음 위치에 JSON으로 저장됩니다.

```text
data/agent_runs/<task-id>.json
```

### 5. 안전장치

- `AGENT_WORKSPACE_DIR` 밖의 파일 접근 차단
- `../` 상위 경로와 Windows 절대경로 차단
- `.env`, 인증키, 인증서 및 비밀 파일 차단
- 파일 삭제, 임의 셸 명령, 관리자 권한 실행 차단
- Python import와 실행 시간·코드·출력 크기 제한
- 기존 파일 수정 전 `data/agent_backups/<task-id>/`에 자동 백업
- 위험 요청은 도구를 실행하지 않고 `failed` 작업으로 기록

> 현재 Python 격리는 애플리케이션 수준입니다. 신뢰할 수 없는 다중 사용자가 접근하는 공개 서버에서는 컨테이너 또는 별도 OS 사용자 수준의 추가 격리가 필요합니다.

### 6. 사용자 인터페이스

- 밝은 챗봇 스타일 Agent 화면
- 데스크톱 아이콘 사이드바
- 최근 작업 고정 패널
- RAG·Agent·Figure Review·평가 화면 이동 메뉴
- 모바일 햄버거 메뉴와 최근 작업 목록
- 에메랄드 포인트 색상의 반응형 디자인

## 기술 스택

| 구분 | 기술 |
|---|---|
| Frontend | Next.js, React, TypeScript, TailwindCSS |
| Backend | FastAPI, Python 3.11 |
| Local LLM | Ollama, Qwen3 8B |
| Vision | Qwen2.5-VL |
| Embedding | BGE-M3 (`BAAI/bge-m3`) |
| Vector DB | ChromaDB persistent client |
| PDF | PyMuPDF, pdfplumber |
| Chart | Plotly, Matplotlib |
| Test/CI | Pytest, Next.js production build, GitHub Actions |

## 프로젝트 구조

```text
AI_Brain/
├─ backend/
│  ├─ app/agents/          # Agent 계획·권한·실행
│  ├─ app/api/             # RAG, Agent, Figure API
│  ├─ app/services/        # 검색·문서·평가 서비스
│  ├─ app/tools/           # 파일·폴더·Python·RAG 검색 도구
│  └─ tests/
├─ frontend/
│  ├─ app/                 # RAG, Agent, Review, Evaluation 화면
│  ├─ components/
│  └─ lib/                 # API 클라이언트
├─ data/
│  ├─ raw/
│  ├─ extracted/
│  ├─ figures/
│  ├─ figure_notes/
│  ├─ vector_db/
│  ├─ metadata/
│  ├─ agent_runs/
│  └─ agent_backups/
└─ workspace/              # Agent가 접근할 수 있는 작업공간
```

## 빠른 시작

### 1. Ollama 모델 준비

Ollama를 설치하고 실행한 뒤 모델을 받습니다.

```bash
ollama pull qwen3:8b
ollama pull qwen2.5vl:7b
```

### 2. 환경 설정

저장소 루트의 `.env.example`을 `.env`로 복사합니다.

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

macOS/Linux:

```bash
cp .env.example .env
```

주요 Agent 설정:

```env
AGENT_WORKSPACE_DIR=./workspace
AGENT_PYTHON_TIMEOUT_SECONDS=30
AGENT_MAX_FILE_BYTES=5242880
```

### 3-A. Docker Compose 실행

```bash
docker compose up --build
```

- Web: <http://localhost:3000>
- Backend API: <http://localhost:8000/api>
- Agent: <http://localhost:3000/agent>

### 3-B. Docker 없이 실행

Windows PowerShell 백엔드:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

macOS/Linux 백엔드:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

새 터미널에서 프론트엔드:

```bash
cd frontend
corepack enable pnpm
pnpm install
pnpm dev
```

시스템 준비 상태 확인:

```bash
curl http://127.0.0.1:8000/api/system/checklist
```

## 테스트

백엔드 Agent 테스트:

```bash
cd backend
pytest -q tests/test_agent_api.py tests/test_agent_tools.py
```

프론트엔드 production build:

```bash
cd frontend
pnpm build
```

Docker E2E:

```bash
python scripts/e2e/run_e2e.py --use-compose
```

로컬 E2E:

```bash
python scripts/e2e/run_local_e2e.py
```

## 현재 상태와 한계

현재 버전은 **안전성을 우선한 규칙 기반 Agent v1**입니다.

- 정해진 파일·CSV 작업은 안정적으로 수행할 수 있습니다.
- 복잡한 자연어 요청을 자유롭게 여러 단계로 나누는 완전 자율 Agent는 아닙니다.
- 파일 삭제, 인터넷 검색, 패키지 설치와 Git 자동 작업은 지원하지 않습니다.
- 공개 서비스 운영 전에는 Python 실행을 OS 또는 컨테이너 수준으로 추가 격리해야 합니다.
- Agent 화면은 새 챗봇 스타일 UI가 적용됐지만 기존 RAG 화면은 아직 같은 디자인으로 통일되지 않았습니다.

## 앞으로의 개발 계획

### 우선순위 1 · UI 통일

- RAG 채팅 화면에 Agent와 같은 공통 사이드바 적용
- 대화 기록 패널 추가
- 중앙 채팅 영역과 하단 고정 입력창 구성
- 출처와 관련 그림을 채팅 카드 형태로 표시
- 모바일 RAG 채팅 화면 개선

### 우선순위 2 · 데이터 분석 확장

- 그래프 제목·축 이름 직접 설정

### 우선순위 3 · 석유공학 전문화

- Porosity, Permeability, BHP, Injection rate 분석
- 석유공학 단위 변환과 추가 계산식 지원
- MRST 결과 형식과 열 별칭 확대
- 생성된 MRST 분석 결과를 RAG 문헌 근거와 자동 비교

### 우선순위 4 · Agent v2

- Ollama 기반 자연어 작업 계획 생성
- 규칙 기반 안전성 검사와 LLM 계획 결합
- 여러 단계 작업의 승인 기반 연속 실행
- 컨테이너 수준 Python 격리
- 전체 RAG·Agent 회귀 테스트 강화

## 한 줄 요약

> AI_Brain은 석유공학 자료에 답변하는 RAG 시스템에서 출발해, 연구 데이터를 분석하고 보고서와 그래프를 안전하게 생성하는 실행형 석유공학 AI Agent로 발전하고 있습니다.
