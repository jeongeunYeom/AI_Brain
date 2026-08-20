from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import textwrap
from uuid import uuid4

from app.agents.permission_manager import PermissionManager
from app.models.agent_schemas import (
    AgentAction,
    AgentPlanRequest,
    AgentToolName,
)
from app.tools.file_tools import FileTools
from app.tools.python_tools import PythonTools


AGENT_SYSTEM_PROMPT = """당신은 석유공학 연구지원 Agent이다.
1. 사용자의 명시적인 요청 범위 안에서만 작업한다.
2. 파일을 수정하거나 코드를 실행하기 전에 작업 계획을 작성한다.
3. 중요한 작업은 사용자의 승인을 받은 뒤 수행한다.
4. 지정된 작업공간 밖의 파일에는 접근하지 않는다.
5. 기존 파일을 수정하기 전에 백업한다.
6. 확인되지 않은 파일이나 데이터 내용을 추측하지 않는다.
7. 실행 결과와 오류를 숨기지 않고 그대로 보고한다.
8. 작업이 실패하면 성공한 것처럼 답변하지 않는다.
9. 최소한의 파일과 명령만 사용한다.
10. 석유공학 계산 결과에는 입력값, 단위와 계산 방법을 표시한다.
"""


_COLUMN_ALIASES = {
    "srco2": (
        "srco2",
        "residual co2 saturation",
        "잔류 co2 포화도",
        "잔류 이산화탄소 포화도",
        "잔류 포화도",
    ),
    "trapped_ratio": (
        "trapped_ratio",
        "trapped ratio",
        "trapped co2 ratio",
        "포획된 co2 비율",
        "포획 co2 비율",
        "포획된 이산화탄소 비율",
        "포획 이산화탄소 비율",
        "포획된 비율",
        "포획 비율",
    ),
    "free_ratio": (
        "free_ratio",
        "free ratio",
        "free co2 ratio",
        "자유 상태 co2 비율",
        "자유 co2 비율",
        "자유 상태 이산화탄소 비율",
        "자유 이산화탄소 비율",
        "자유 상태 비율",
        "자유 비율",
    ),
    "total_storage_mt": (
        "total_storage_mt",
        "total storage",
        "총 저장량",
        "전체 저장량",
    ),
}


@dataclass
class PlannedTask:
    plan: list[str]
    actions: list[AgentAction]


class AgentPlanner:
    def __init__(
        self,
        permissions: PermissionManager,
        file_tools: FileTools,
        python_tools: PythonTools,
    ):
        self.permissions = permissions
        self.file_tools = file_tools
        self.python_tools = python_tools

    def plan(self, request: AgentPlanRequest) -> PlannedTask:
        text = request.request.strip()
        lowered = text.lower()
        target = self._resolve_target(request.target_path, text)
        output = request.output_path

        if request.python_code:
            code = request.python_code.strip()
            self.python_tools.validate(code)
            return PlannedTask(
                plan=[
                    "실행할 Python 코드를 안전성 규칙으로 검사합니다.",
                    "승인 후 제한된 Python 프로세스에서 코드를 실행합니다.",
                    "표준 출력, 오류 및 생성 파일을 작업 기록에 저장합니다.",
                ],
                actions=[
                    self._action(
                        AgentToolName.RUN_PYTHON,
                        "제공된 Python 코드를 제한된 환경에서 실행",
                        {"code": code},
                        requires_approval=True,
                        preview=code,
                    )
                ],
            )

        if self._looks_like_edit(lowered):
            if not target:
                raise ValueError("파일 수정 작업에는 target_path가 필요합니다.")
            old_text, new_text = self._replacement_values(request, text)
            if old_text is None or new_text is None:
                raise ValueError(
                    "부분 수정에는 old_text와 new_text를 입력하거나, "
                    "요청에 '기존문구 -> 새문구' 형식으로 작성해야 합니다."
                )
            preview = self.file_tools.preview_edit(target, old_text, new_text)["diff"]
            return PlannedTask(
                plan=[
                    f"{target} 파일의 현재 내용을 확인합니다.",
                    "변경 전후 차이를 검토합니다.",
                    "승인 후 원본을 백업하고 지정된 한 부분만 수정합니다.",
                    "백업 경로와 변경 내용을 작업 기록에 저장합니다.",
                ],
                actions=[
                    self._action(
                        AgentToolName.READ_FILE,
                        f"수정 대상 파일 확인: {target}",
                        {"path": target, "start_line": 1, "end_line": 500},
                        target_files=[target],
                    ),
                    self._action(
                        AgentToolName.EDIT_FILE,
                        f"{target}의 지정된 내용만 수정",
                        {"path": target, "old_text": old_text, "new_text": new_text},
                        target_files=[target],
                        requires_approval=True,
                        preview=preview,
                    ),
                ],
            )

        if self._looks_like_scatter(lowered):
            csv_path = self._require_csv_target(target)
            output_path = output or "results/scatter_plot.png"
            self.permissions.resolve_path(
                output_path,
                must_exist=False,
                allow_directory=False,
                allowed_extensions={".png"},
            )
            csv_info = self._csv_info(csv_path)
            x_name, y_name = self._select_scatter_columns(request, text, csv_info)
            code = self._scatter_code(csv_path, output_path, x_name, y_name)
            self.python_tools.validate(code)
            return PlannedTask(
                plan=[
                    f"{csv_path}의 열 구조와 데이터 개수를 확인합니다.",
                    f"X축은 {x_name}, Y축은 {y_name} 열로 확정합니다.",
                    "두 열의 숫자 데이터에서 결측값과 변환 불가능한 행을 제외합니다.",
                    "승인 후 Python으로 산점도를 생성합니다.",
                    f"그래프를 {output_path}에 저장하고 결과를 검증합니다.",
                ],
                actions=[
                    self._action(
                        AgentToolName.READ_FILE,
                        f"CSV 구조 확인: {csv_path}",
                        {"path": csv_path, "start_line": 1, "end_line": 25},
                        target_files=[csv_path],
                    ),
                    self._action(
                        AgentToolName.RUN_PYTHON,
                        f"{x_name}와 {y_name} 열의 산점도 생성",
                        {
                            "code": code,
                            "x_column": x_name,
                            "y_column": y_name,
                            "expected_outputs": [output_path],
                        },
                        target_files=[csv_path, output_path],
                        requires_approval=True,
                        preview=code,
                    ),
                ],
            )

        if self._looks_like_csv_report(lowered):
            csv_path = self._require_csv_target(target)
            output_path = output or "results/csv_analysis_report.txt"
            self.permissions.resolve_path(
                output_path,
                must_exist=False,
                allow_directory=False,
                allowed_extensions={".txt"},
            )
            code = self._csv_report_code(csv_path, output_path)
            self.python_tools.validate(code)
            return PlannedTask(
                plan=[
                    f"{csv_path}의 열 구조와 데이터 개수를 확인합니다.",
                    "숫자 열의 개수, 결측값, 최소·최대·평균을 계산합니다.",
                    "승인 후 분석 코드를 실행합니다.",
                    f"분석 보고서를 {output_path}에 저장하고 내용을 검증합니다.",
                ],
                actions=[
                    self._action(
                        AgentToolName.READ_FILE,
                        f"CSV 구조 확인: {csv_path}",
                        {"path": csv_path, "start_line": 1, "end_line": 25},
                        target_files=[csv_path],
                    ),
                    self._action(
                        AgentToolName.RUN_PYTHON,
                        "CSV 통계 보고서 생성",
                        {
                            "code": code,
                            "expected_outputs": [output_path],
                        },
                        target_files=[csv_path, output_path],
                        requires_approval=True,
                        preview=code,
                    ),
                ],
            )

        if request.content is not None or self._looks_like_create(lowered):
            output_path = output or target or "results/agent_output.txt"
            if request.content is None:
                raise ValueError(
                    "새 파일 생성에는 content를 입력해야 합니다. "
                    "분석 결과 자동 생성은 CSV 분석 요청으로 작성하세요."
                )
            self.permissions.resolve_path(
                output_path,
                must_exist=False,
                allow_directory=False,
            )
            return PlannedTask(
                plan=[
                    f"생성할 파일 경로 {output_path}가 작업공간 내부인지 확인합니다.",
                    "동일한 파일이 이미 있는지 확인합니다.",
                    "승인 후 새 파일을 생성합니다. 기존 파일은 덮어쓰지 않습니다.",
                    "생성된 파일의 존재 여부와 내용을 검증합니다.",
                ],
                actions=[
                    self._action(
                        AgentToolName.CREATE_FILE,
                        f"새 결과 파일 생성: {output_path}",
                        {"path": output_path, "content": request.content},
                        target_files=[output_path],
                        requires_approval=True,
                        preview=request.content[:4000],
                    )
                ],
            )

        if self._looks_like_csv_structure(lowered):
            csv_path = self._require_csv_target(target)
            return PlannedTask(
                plan=[
                    f"{csv_path} 파일을 읽기 전용으로 엽니다.",
                    "CSV 열 이름, 데이터 행 수와 일부 샘플을 확인합니다.",
                    "파일 변경 없이 구조 정보를 반환합니다.",
                ],
                actions=[
                    self._action(
                        AgentToolName.READ_FILE,
                        f"CSV 파일 구조 확인: {csv_path}",
                        {"path": csv_path, "start_line": 1, "end_line": 25},
                        target_files=[csv_path],
                    )
                ],
            )

        if self._looks_like_figure_search(lowered):
            return PlannedTask(
                plan=[
                    "질문에서 찾을 Figure 주제와 핵심 용어를 확인합니다.",
                    "Figure Note와 관련 문서 근거를 읽기 전용으로 검색합니다.",
                    "관련 Figure 원본 경로와 문서·페이지 정보를 반환합니다.",
                ],
                actions=[
                    self._action(
                        AgentToolName.GET_RELATED_FIGURES,
                        "관련 Figure와 문서 근거 검색",
                        {"query": text, "top_k": 5},
                    )
                ],
            )

        if self._looks_like_knowledge_search(lowered):
            return PlannedTask(
                plan=[
                    "질문에서 문헌 검색에 사용할 핵심 용어를 확인합니다.",
                    "Text/Figure RAG 지식베이스를 읽기 전용으로 검색합니다.",
                    "검색된 문서·페이지·근거 문장을 반환합니다.",
                ],
                actions=[
                    self._action(
                        AgentToolName.SEARCH_KNOWLEDGE_BASE,
                        "관련 문헌과 근거 검색",
                        {"query": text, "top_k": 5},
                    )
                ],
            )

        if target and self._looks_like_read(lowered):
            return PlannedTask(
                plan=[
                    f"{target} 경로가 작업공간 내부인지 확인합니다.",
                    "파일을 최대 500줄 범위에서 읽기 전용으로 확인합니다.",
                    "파일 내용과 메타데이터를 반환합니다.",
                ],
                actions=[
                    self._action(
                        AgentToolName.READ_FILE,
                        f"파일 읽기: {target}",
                        {"path": target, "start_line": 1, "end_line": 500},
                        target_files=[target],
                    )
                ],
            )

        directory = target or "."
        return PlannedTask(
            plan=[
                f"{directory} 경로가 작업공간 내부인지 확인합니다.",
                "하위 파일과 폴더의 이름, 형식, 크기와 수정 시간을 읽습니다.",
                "파일을 변경하지 않고 목록을 반환합니다.",
            ],
            actions=[
                self._action(
                    AgentToolName.LIST_DIRECTORY,
                    f"폴더 목록 확인: {directory}",
                    {"path": directory},
                    target_files=[directory],
                )
            ],
        )

    def _resolve_target(self, explicit: str | None, text: str) -> str | None:
        candidate = explicit.strip() if explicit else self._extract_path(text)
        if not candidate:
            return None
        path = self.permissions.resolve_path(
            candidate,
            must_exist=False,
            allow_directory=True,
        )
        return self.permissions.to_relative(path) or "."

    def _require_csv_target(self, target: str | None) -> str:
        if target:
            path = self.permissions.resolve_path(
                target,
                must_exist=True,
                allow_directory=False,
                allowed_extensions={".csv"},
            )
        else:
            path = self.permissions.find_first_file(".csv")
        return self.permissions.to_relative(path)

    def _csv_info(self, csv_path: str) -> dict:
        result = self.file_tools.read_file(csv_path, start_line=1, end_line=6)
        csv_info = result.get("csv")
        if not csv_info or not csv_info.get("columns"):
            raise ValueError("CSV에 열 이름이 없습니다.")
        if csv_info.get("row_count", 0) == 0:
            raise ValueError("CSV에 데이터 행이 없습니다.")
        return csv_info

    def _select_scatter_columns(
        self,
        request: AgentPlanRequest,
        text: str,
        csv_info: dict,
    ) -> tuple[str, str]:
        columns = [str(column) for column in csv_info["columns"]]
        explicit_x = (request.x_column or "").strip()
        explicit_y = (request.y_column or "").strip()

        if bool(explicit_x) != bool(explicit_y):
            raise ValueError("산점도에는 X축과 Y축 열을 모두 선택해야 합니다.")

        if explicit_x and explicit_y:
            x_name = self._resolve_column_name(explicit_x, columns)
            y_name = self._resolve_column_name(explicit_y, columns)
            self._validate_distinct_columns(x_name, y_name)
            self._validate_numeric_samples(x_name, y_name, csv_info)
            return x_name, y_name

        mentioned = self._mentioned_columns(text, columns)
        if len(mentioned) >= 2:
            x_name, y_name = mentioned[:2]
            self._validate_distinct_columns(x_name, y_name)
            self._validate_numeric_samples(x_name, y_name, csv_info)
            return x_name, y_name

        numeric = self._numeric_sample_columns(csv_info)
        if len(numeric) < 2:
            raise ValueError(
                "산점도에 사용할 숫자 열이 2개 이상 필요합니다. "
                f"사용 가능한 열: {', '.join(columns)}"
            )
        return numeric[0], numeric[1]

    @staticmethod
    def _resolve_column_name(requested: str, columns: list[str]) -> str:
        lookup = {column.casefold(): column for column in columns}
        matched = lookup.get(requested.casefold())
        if matched is None:
            raise ValueError(
                f"CSV에 '{requested}' 열이 없습니다. "
                f"사용 가능한 열: {', '.join(columns)}"
            )
        return matched

    @classmethod
    def _mentioned_columns(cls, text: str, columns: list[str]) -> list[str]:
        normalized_text = cls._normalize_column_text(text)
        matches: list[tuple[int, str]] = []
        for column in columns:
            positions: list[int] = []
            candidates = {
                column,
                column.replace("_", " "),
                column.replace("_", ""),
            }
            candidates.update(_COLUMN_ALIASES.get(column.casefold(), ()))
            for candidate in candidates:
                normalized_candidate = cls._normalize_column_text(candidate)
                if not normalized_candidate:
                    continue
                position = normalized_text.find(normalized_candidate)
                if position >= 0:
                    positions.append(position)
            if positions:
                matches.append((min(positions), column))
        matches.sort(key=lambda item: item[0])
        return [column for _, column in matches]

    @staticmethod
    def _normalize_column_text(value: str) -> str:
        lowered = value.casefold().replace("co₂", "co2")
        return re.sub(r"[\s_\-]+", " ", lowered).strip()

    @staticmethod
    def _numeric_sample_columns(csv_info: dict) -> list[str]:
        columns = [str(column) for column in csv_info["columns"]]
        sample_rows = csv_info.get("sample_rows") or []
        numeric: list[str] = []
        for index, column in enumerate(columns):
            values = 0
            for row in sample_rows:
                if index >= len(row):
                    continue
                raw = str(row[index]).strip()
                if not raw:
                    continue
                try:
                    float(raw)
                except ValueError:
                    continue
                values += 1
            if values > 0:
                numeric.append(column)
        return numeric

    @classmethod
    def _validate_numeric_samples(
        cls,
        x_name: str,
        y_name: str,
        csv_info: dict,
    ) -> None:
        numeric = set(cls._numeric_sample_columns(csv_info))
        invalid = [name for name in (x_name, y_name) if name not in numeric]
        if invalid:
            raise ValueError(
                "선택한 열에서 숫자 데이터를 확인할 수 없습니다: "
                + ", ".join(invalid)
            )

    @staticmethod
    def _validate_distinct_columns(x_name: str, y_name: str) -> None:
        if x_name == y_name:
            raise ValueError("X축과 Y축에는 서로 다른 열을 선택해야 합니다.")

    @staticmethod
    def _extract_path(text: str) -> str | None:
        quoted = re.findall(r"[\"'`](.+?)[\"'`]", text)
        for value in quoted:
            if Path(value).suffix.lower() in {
                ".txt", ".md", ".py", ".json", ".csv", ".yaml", ".yml", ".log", ".png"
            }:
                return value.strip()

        match = re.search(
            r"([\w가-힣.()\-\\/]+\.(?:txt|md|py|json|csv|ya?ml|log|png))",
            text,
            flags=re.IGNORECASE,
        )
        return match.group(1).strip() if match else None

    @staticmethod
    def _replacement_values(
        request: AgentPlanRequest,
        text: str,
    ) -> tuple[str | None, str | None]:
        if request.old_text is not None and request.new_text is not None:
            return request.old_text, request.new_text
        match = re.search(r"[\"'](.+?)[\"']\s*(?:을|를)?\s*[-=]>\s*[\"'](.+?)[\"']", text)
        if match:
            return match.group(1), match.group(2)
        return None, None

    @staticmethod
    def _looks_like_edit(text: str) -> bool:
        return any(word in text for word in ("수정", "변경", "바꿔", "replace", "edit"))

    @staticmethod
    def _looks_like_create(text: str) -> bool:
        return any(word in text for word in ("파일 생성", "파일로 저장", "작성해", "create file"))

    @staticmethod
    def _looks_like_read(text: str) -> bool:
        return any(word in text for word in ("읽", "내용", "확인", "설명", "read"))

    @staticmethod
    def _looks_like_csv_structure(text: str) -> bool:
        return "csv" in text and any(
            word in text for word in ("구조", "열", "column", "행 수", "데이터 개수")
        )

    @staticmethod
    def _looks_like_csv_report(text: str) -> bool:
        return "csv" in text and any(
            word in text for word in ("분석", "통계", "보고서", "report")
        ) and any(word in text for word in ("저장", "파일", "report", "보고서"))

    @staticmethod
    def _looks_like_scatter(text: str) -> bool:
        return "csv" in text and any(
            word in text for word in ("산점도", "scatter", "그래프", "plot")
        )

    @staticmethod
    def _looks_like_figure_search(text: str) -> bool:
        figure_terms = (
            "figure",
            "그래프",
            "도표",
            "그림",
            "plot",
            "chart",
        )
        search_terms = (
            "찾",
            "검색",
            "관련",
            "근거",
            "문헌",
            "논문",
            "search",
        )
        return any(term in text for term in figure_terms) and any(
            term in text for term in search_terms
        )

    @staticmethod
    def _looks_like_knowledge_search(text: str) -> bool:
        knowledge_terms = (
            "문헌",
            "논문",
            "이론",
            "지식베이스",
            "rag",
            "문서 근거",
            "출처",
            "교재",
            "literature",
            "knowledge base",
        )
        return any(term in text for term in knowledge_terms)

    def _action(
        self,
        tool: AgentToolName,
        description: str,
        arguments: dict,
        *,
        target_files: list[str] | None = None,
        requires_approval: bool = False,
        preview: str | None = None,
    ) -> AgentAction:
        return AgentAction(
            action_id=f"A-{uuid4().hex[:8]}",
            tool=tool,
            description=description,
            arguments=arguments,
            target_files=target_files or [],
            requires_approval=requires_approval,
            preview=preview,
        )

    @staticmethod
    def _csv_report_code(csv_path: str, output_path: str) -> str:
        return textwrap.dedent(
            f"""
            import csv
            import statistics

            input_path = {csv_path!r}
            output_path = {output_path!r}

            with open(input_path, "r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))

            columns = list(rows[0].keys()) if rows else []
            lines = [
                "CSV 분석 보고서",
                f"입력 파일: {{input_path}}",
                f"데이터 행 수: {{len(rows)}}",
                f"열: {{', '.join(columns) if columns else '(없음)'}}",
                "",
            ]

            for column in columns:
                values = []
                missing = 0
                for row in rows:
                    raw = (row.get(column) or "").strip()
                    if not raw:
                        missing += 1
                        continue
                    try:
                        values.append(float(raw))
                    except ValueError:
                        pass
                lines.append(f"[{{column}}]")
                lines.append(f"- 결측값: {{missing}}")
                if values:
                    lines.append(f"- 숫자 데이터 수: {{len(values)}}")
                    lines.append(f"- 최소: {{min(values):.6g}}")
                    lines.append(f"- 최대: {{max(values):.6g}}")
                    lines.append(f"- 평균: {{statistics.fmean(values):.6g}}")
                else:
                    lines.append("- 숫자 열이 아님")
                lines.append("")

            with open(output_path, "w", encoding="utf-8") as handle:
                handle.write(chr(10).join(lines))

            print(f"Saved report: {{output_path}}")
            """
        ).strip()

    @staticmethod
    def _scatter_code(
        csv_path: str,
        output_path: str,
        x_name: str,
        y_name: str,
    ) -> str:
        return textwrap.dedent(
            f"""
            import csv
            import matplotlib.pyplot as plt

            input_path = {csv_path!r}
            output_path = {output_path!r}
            x_name = {x_name!r}
            y_name = {y_name!r}

            with open(input_path, "r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))

            if not rows:
                raise ValueError("CSV에 데이터 행이 없습니다.")

            columns = list(rows[0].keys())
            missing_columns = [name for name in (x_name, y_name) if name not in columns]
            if missing_columns:
                raise ValueError("CSV 열을 찾을 수 없습니다: " + ", ".join(missing_columns))

            x_values = []
            y_values = []
            for row in rows:
                try:
                    x_value = float((row.get(x_name) or "").strip())
                    y_value = float((row.get(y_name) or "").strip())
                except ValueError:
                    continue
                x_values.append(x_value)
                y_values.append(y_value)

            if not x_values:
                raise ValueError("선택한 두 열에 함께 사용할 수 있는 숫자 행이 없습니다.")

            plt.figure(figsize=(8, 5))
            plt.scatter(x_values, y_values)
            plt.xlabel(x_name)
            plt.ylabel(y_name)
            plt.title(f"{{y_name}} vs {{x_name}}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            with open(output_path, "wb"):
                pass
            plt.savefig(output_path, dpi=160)
            plt.close()
            print(f"Saved scatter plot: {{output_path}}")
            """
        ).strip()
