# Well Test Agent MVP 작업 순서

## 오늘 목표

1. 현재 상태를 별도 커밋으로 보존
2. `codex/welltest-agent-mvp` 브랜치 생성
3. 16개 benchmark 질문 추가
4. Well Test 공학 검증 모듈과 단위 테스트 추가
5. QA 답변 생성 뒤 최대 2회 재작성
6. `data/agent_runs`에 질문별 JSON 기록
7. 4개 smoke test 후 전체 benchmark 실행

## 중요한 수정점

초기 제안처럼 `validate_well_test_answer(answer)`만 받으면 부족합니다.

반드시 다음을 같이 받아야 합니다.

```python
validate_well_test_answer(
    question,
    answer,
    retrieved_sources=sources,
)
```

이유:

- 나쁜 답변이 radial flow 자체를 빼면 필수 규칙 검사를 피할 수 있음
- 검색 결과가 0개인지 알아야 정확한 거절을 검사할 수 있음
- “따르지 않는다” 같은 올바른 부정문을 금지 패턴으로 오탐하면 안 됨

## QA 통합 흐름

```text
검색 1회
→ 동일한 근거 prompt 생성
→ 초안 생성
→ EngineeringValidator 검사
→ 실패 시 같은 근거로 재작성 1
→ 실패 시 같은 근거로 재작성 2
→ 계속 실패하면 검토 필요 응답
→ 전체 실행 기록을 원자적으로 JSON 저장
```

초안 1회 + 재작성 2회이므로 총 생성 횟수는 최대 3회입니다.

## 재작성 프롬프트

```text
이전 답변에서 다음 공학 검증 오류가 발견되었습니다.

{errors}

동일한 검색 근거만 사용하여 답변을 다시 작성하세요.
새로운 사실이나 수치를 추가하지 마세요.

필수 조건:
1. Wellbore storage에서는 pressure와 pressure derivative가 겹치며
   unit-slope diagonal을 따른다고 설명합니다.
2. Radial flow에서는 pressure derivative가 plateau를 형성한다고
   설명합니다.
3. 서로 다른 Figure의 설명을 혼합하지 않습니다.
4. 문서에 없는 내용은 추가하지 않습니다.
5. 모든 사실 문장에 문서명과 페이지를 표시합니다.
```

## Agent run JSON에 추가할 값

오류만 저장하지 말고 다음도 남겨야 논문 지표를 계산할 수 있습니다.

- benchmark_id
- prompt_version
- 검색 문서, page, chunk_id, score
- 각 attempt의 실제 answer
- attempt별 생성 시간
- validation rule id
- 최종 상태
- 전체 실행 시간

## 최소 완료 조건

- known bad radial-flow 문장을 차단
- 부정형 수정문은 오탐하지 않음
- 검색 결과가 0개면 고정 거절문만 허용
- 질문당 JSON 1개 생성
- 검색은 1회만 수행
- 최대 생성 횟수 3회
- ChromaDB와 Figure Note는 변경하지 않음
