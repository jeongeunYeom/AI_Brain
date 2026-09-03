# petroleum-rag-agent · Petroleum Engineering RAG Agent

석유공학 문서에 근거해 답변하는 RAG와 연구 파일을 직접 분석하는 Agent를 결합한 로컬 우선 웹 애플리케이션입니다.

petroleum-rag-agent의 목표는 단순한 질의응답을 넘어 PDF와 연구 데이터를 검색·분석하고, 보고서와 그래프를 안전하게 생성하는 **실행형 석유공학 연구지원 AI**를 만드는 것입니다.

## 핵심 목표

- 석유공학 PDF와 교재를 기반으로 출처가 있는 답변 생성
- CSV와 연구 결과를 분석하여 통계·보고서·그래프 생성
- 사용자가 작업 계획과 실행 코드를 확인한 뒤 승인
- 로컬 Ollama 모델을 사용하여 외부 API 비용 최소화
- 모든 파일 작업을 제한된 workspace 안에서 수행하고 기록
- 석유공학 문서 검색과 연구 데이터 분석 도구로 확장

## 현재 제공 기능

### 1. 문서 기반 RAG

- PDF 업로드, 텍스트 추출 및 청크 분할
- BGE-M3 임베딩과 ChromaDB 영구 저장
- 질문 유형에 따른 검색 전략 선택
- 검색 문서에 근거한 답변과 문서·페이지 출처 표시
- Ontology v0.1 concept metadata와 간단한 relation graph 생성
- 문서별 relation graph를 `data/ontology/<document_id>.jsonl`에 저장
- 업로드 완료 후 문서 목록과 Documents/Chunks 상태 자동 갱신
- RAG 채팅 기록은 새 메시지가 생긴 경우에만 최근 대화 순서 갱신
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
- 빈 새 대화는 기록에 저장하지 않고, 새 요청으로 계획이 생성될 때 대화 목록 갱신

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

### 3. 결과 미리보기와 다운로드

- PNG 그래프 미리보기 및 다운로드
- TXT·Markdown 본문 미리보기 및 다운로드
- CSV 표 미리보기 및 다운로드
- 생성 결과를 Agent 화면에서 바로 확인

### 4. 작업 기록

- 하나의 Agent 대화에 여러 작업 누적
- 새 요청이 있는 대화 생성, 과거 대화 목록 및 대화 전체 복원
- 대화별 작업 기록 연결
- 같은 대화에서 `아까 결과`, `같은 파일`처럼 요청하면 이전 완료 작업의 파일 자동 참조
- 자동 참조한 원본 작업 ID와 파일 경로를 계획 화면에 표시
- 최근 Agent 작업 50개를 최신순으로 표시
- `completed`, `failed`, `canceled` 상태 확인
- 요청 내용, 생성 시각과 실행 시간 표시
- 사용 도구와 생성·수정 결과 확인
- 작업 상세 기록 다시 열기
- 손상된 기록 파일은 전체 화면을 중단하지 않고 제외

작업 기록은 다음 위치에 JSON으로 저장됩니다.

```text
data/agent_runs/<task-id>.json
data/agent_conversations/<conversation-id>.json
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
- 데스크톱 아이콘 레일과 RAG/Agent 확장 사이드바 고정 레이아웃
- RAG 화면의 문서 목록과 시스템 상태 자동 갱신
- 새 대화 내용이 없을 때 RAG/Agent 대화 목록 순서 유지
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
petroleum-rag-agent/
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
│  ├─ ontology/
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

## 논문용 Benchmark v1

`evaluation/well_test_agent_benchmark.json`에는 실제 Well Test 문서 근거로
검증하는 32개 질문이 들어 있습니다. 16개 핵심 개념마다 표현을 바꾼 질문을
한 개씩 추가했으며 구성은 Text 14개, Figure 12개, Hallucination 6개입니다.
따라서 논문에서는 **32개 독립 개념**이 아니라 **16개 개념·32개 질문**으로
표기하고, 데이터 분할 시 같은 `concept_group`의 질문을 서로 다른 split에
나누지 않아야 합니다.

질문 세트와 실행 조건만 확인:

```bash
python backend/scripts/run_well_test_benchmark.py --dry-run
```

Qwen3 단독과 Qwen3 + RAG 비교:

```bash
python backend/scripts/run_well_test_benchmark.py \
  --mode ollama-direct --model qwen3:8b --condition qwen3_baseline

python backend/scripts/run_well_test_benchmark.py \
  --mode rag --model qwen3:8b --condition qwen3_rag
```

동일 RAG에서 Gemma4 비교:

```bash
python backend/scripts/run_well_test_benchmark.py \
  --mode rag --model gemma4:latest --condition gemma4_rag
```

각 실행이 만든 JSON을 논문용 비교 CSV/JSON으로 통합:

```bash
python backend/scripts/compare_benchmark_runs.py \
  data/evaluation/well_test_benchmark_<baseline-run-id>.json \
  data/evaluation/well_test_benchmark_<qwen-rag-run-id>.json \
  data/evaluation/well_test_benchmark_<gemma-rag-run-id>.json
```

결과에는 답변 정확도, 환각률, 정확한 거절률, 문서·페이지 Retrieval
Recall@K, Figure 답변·검색 정확도와 평균 응답 시간이 기록됩니다. 비교표의
delta는 첫 번째 조건을 기준으로 한 기술 통계이며 통계적 유의성을 의미하지
않습니다. Source별 관련성 정답표가 필요한 Citation Precision은 아직 자동
계산하지 않으므로 후속 수동 라벨링 단계에서 추가합니다.

## 현재 상태와 한계

현재 버전은 **안전성을 우선한 규칙 기반 Agent v1**입니다.

- 정해진 파일·CSV 작업은 안정적으로 수행할 수 있습니다.
- 복잡한 자연어 요청을 자유롭게 여러 단계로 나누는 완전 자율 Agent는 아닙니다.
- 파일 삭제, 인터넷 검색, 패키지 설치와 Git 자동 작업은 지원하지 않습니다.
- 공개 서비스 운영 전에는 Python 실행을 OS 또는 컨테이너 수준으로 추가 격리해야 합니다.
- RAG와 Agent 화면은 데스크톱에서 확장 사이드바를 고정해 사용하는 구조로 통일되어 있습니다.

## 앞으로의 개발 계획

### 우선순위 1 · UI 통일

- 출처와 관련 그림을 채팅 카드 형태로 표시
- 모바일 RAG·Agent 화면 세부 UX 개선
- 사이드바 문서·작업 목록 검색과 필터 강화

### 우선순위 2 · 데이터 분석 확장

- 그래프 제목·축 이름 직접 설정

### 우선순위 3 · 석유공학 전문화

- Porosity, Permeability, BHP, Injection rate 분석
- 석유공학 단위 변환과 추가 계산식 지원
- Well test, reservoir, drilling 관련 열 별칭 확대
- 생성된 분석 결과를 RAG 문헌 근거와 자동 비교

### 우선순위 4 · Agent v2

- Ollama 기반 자연어 작업 계획 생성
- 규칙 기반 안전성 검사와 LLM 계획 결합
- 여러 단계 작업의 승인 기반 연속 실행
- 컨테이너 수준 Python 격리
- 전체 RAG·Agent 회귀 테스트 강화

## 한 줄 요약

> petroleum-rag-agent는 석유공학 자료에 답변하는 RAG 시스템에서 출발해, 연구 데이터를 분석하고 보고서와 그래프를 안전하게 생성하는 실행형 석유공학 AI Agent로 발전하고 있습니다.
