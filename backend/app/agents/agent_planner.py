from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import textwrap
from uuid import uuid4

from app.agents.permission_manager import PermissionManager
from app.models.agent_schemas import (
    AgentAction,
    AgentAnalysisProfile,
    AgentChartType,
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

_MRST_COLUMN_ALIASES = {
    "srco2": _COLUMN_ALIASES["srco2"] + (
        "sr_co2",
        "residual gas saturation",
        "residual saturation",
    ),
    "time": (
        "time",
        "time_years",
        "simulation_time",
        "simulation_time_years",
        "year",
        "years",
        "시간",
        "시간 년",
        "시뮬레이션 시간",
    ),
    "trapped_ratio": _COLUMN_ALIASES["trapped_ratio"] + (
        "trapped_co2_fraction",
        "trapped_fraction",
        "immobilized_co2_ratio",
    ),
    "free_ratio": _COLUMN_ALIASES["free_ratio"] + (
        "free_co2_fraction",
        "free_fraction",
        "mobile_co2_ratio",
        "mobile_ratio",
    ),
    "trapped_amount": (
        "trapped_co2",
        "trapped_co2_mt",
        "trapped_mass",
        "trapped_mass_mt",
        "trapped_volume",
        "immobilized_co2",
        "포획 co2 양",
        "포획 co2 질량",
        "포획량",
    ),
    "free_amount": (
        "free_co2",
        "free_co2_mt",
        "free_mass",
        "free_mass_mt",
        "free_volume",
        "mobile_co2",
        "자유 co2 양",
        "자유 co2 질량",
        "자유량",
    ),
    "total_storage": _COLUMN_ALIASES["total_storage_mt"] + (
        "total_co2",
        "total_co2_mt",
        "stored_co2",
        "stored_co2_mt",
        "total_storage_mass",
        "총 co2 저장량",
    ),
}

_CHART_LABELS = {
    AgentChartType.SCATTER: "산점도",
    AgentChartType.LINE: "선 그래프",
    AgentChartType.BAR: "막대그래프",
    AgentChartType.HISTOGRAM: "히스토그램",
}

_CHART_DEFAULT_OUTPUTS = {
    AgentChartType.SCATTER: "results/scatter_plot.png",
    AgentChartType.LINE: "results/line_plot.png",
    AgentChartType.BAR: "results/bar_chart.png",
    AgentChartType.HISTOGRAM: "results/histogram.png",
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
        targets = self._resolve_targets(request.target_paths, text)
        if target and target not in targets:
            targets.insert(0, target)
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

        if self._looks_like_mrst_co2_analysis(request, lowered, target, targets):
            return self._plan_mrst_co2_analysis(
                text,
                targets or ([target] if target else []),
                output,
            )

        if self._looks_like_multi_csv_compare(request, lowered, targets):
            return self._plan_multi_csv_comparison(
                request,
                text,
                targets,
                output,
            )

        chart_type = self._requested_chart_type(request, lowered, target)
        if chart_type is not None:
            return self._plan_csv_chart(
                request,
                text,
                target,
                output,
                chart_type,
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
                    "숫자 열의 최소·최대·평균·중앙값·표준편차를 계산합니다.",
                    "숫자 열 조합의 Pearson 상관계수를 계산합니다.",
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

    def _resolve_targets(self, explicit: list[str], text: str) -> list[str]:
        candidates = explicit or [
            path for path in self._extract_paths(text) if Path(path).suffix.lower() == ".csv"
        ]
        resolved: list[str] = []
        for candidate in candidates:
            path = self.permissions.resolve_path(
                candidate.strip(),
                must_exist=False,
                allow_directory=False,
                allowed_extensions={".csv"},
            )
            relative = self.permissions.to_relative(path)
            if relative not in resolved:
                resolved.append(relative)
        return resolved

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

    def _require_csv_targets(self, targets: list[str]) -> list[str]:
        if len(targets) < 2:
            raise ValueError("CSV 비교에는 서로 다른 CSV 파일이 2개 이상 필요합니다.")
        resolved: list[str] = []
        for target in targets:
            path = self.permissions.resolve_path(
                target,
                must_exist=True,
                allow_directory=False,
                allowed_extensions={".csv"},
            )
            relative = self.permissions.to_relative(path)
            if relative not in resolved:
                resolved.append(relative)
        if len(resolved) < 2:
            raise ValueError("CSV 비교에는 서로 다른 CSV 파일이 2개 이상 필요합니다.")
        return resolved

    def _csv_info(self, csv_path: str) -> dict:
        result = self.file_tools.read_file(csv_path, start_line=1, end_line=6)
        csv_info = result.get("csv")
        if not csv_info or not csv_info.get("columns"):
            raise ValueError("CSV에 열 이름이 없습니다.")
        if csv_info.get("row_count", 0) == 0:
            raise ValueError("CSV에 데이터 행이 없습니다.")
        return csv_info

    def _plan_mrst_co2_analysis(
        self,
        text: str,
        targets: list[str],
        output: str | None,
    ) -> PlannedTask:
        csv_paths = self._require_mrst_csv_targets(targets)
        requested_output = output or "results/mrst_co2_analysis.md"
        output_file = self.permissions.resolve_path(
            requested_output,
            must_exist=False,
            allow_directory=False,
            allowed_extensions={".md", ".csv", ".png"},
        )
        base_path = output_file.with_suffix("")
        report_path = self.permissions.to_relative(base_path.with_suffix(".md"))
        summary_path = self.permissions.to_relative(base_path.with_suffix(".csv"))
        chart_path = self.permissions.to_relative(base_path.with_suffix(".png"))

        column_maps: dict[str, dict[str, str]] = {}
        condition_fallbacks: dict[str, float | None] = {}
        for csv_path in csv_paths:
            csv_info = self._csv_info(csv_path)
            columns = [str(column) for column in csv_info["columns"]]
            column_map = self._match_mrst_columns(columns)
            has_ratio = bool(
                column_map.get("trapped_ratio") or column_map.get("free_ratio")
            )
            has_amounts = bool(
                column_map.get("trapped_amount") and column_map.get("free_amount")
            )
            if not has_ratio and not has_amounts:
                raise ValueError(
                    f"{csv_path}에서 trapped/free 비율 열 또는 "
                    "trapped/free 양 열 쌍을 찾을 수 없습니다. "
                    f"사용 가능한 열: {', '.join(columns)}"
                )
            column_maps[csv_path] = column_map
            condition_fallbacks[csv_path] = self._srco2_from_filename(csv_path)

        code = self._mrst_co2_analysis_code(
            csv_paths,
            column_maps,
            condition_fallbacks,
            summary_path,
            chart_path,
            report_path,
        )
        self.python_tools.validate(code)
        read_actions = [
            self._action(
                AgentToolName.READ_FILE,
                f"MRST 결과 CSV 구조 확인: {path}",
                {"path": path, "start_line": 1, "end_line": 25},
                target_files=[path],
            )
            for path in csv_paths
        ]
        run_action = self._action(
            AgentToolName.RUN_PYTHON,
            f"{len(csv_paths)}개 MRST/CO₂ 결과 CSV 분석",
            {
                "code": code,
                "analysis_profile": AgentAnalysisProfile.MRST_CO2.value,
                "input_paths": csv_paths,
                "column_maps": column_maps,
                "expected_outputs": [summary_path, chart_path, report_path],
            },
            target_files=[
                *csv_paths,
                summary_path,
                chart_path,
                report_path,
            ],
            requires_approval=True,
            preview=code,
        )
        mapped_descriptions = [
            f"{path}: "
            + ", ".join(
                f"{meaning}={column}" for meaning, column in column_maps[path].items()
            )
            for path in csv_paths
        ]
        return PlannedTask(
            plan=[
                f"{len(csv_paths)}개 원본 CSV를 변경하지 않고 구조와 데이터 행을 확인합니다.",
                "MRST/CO₂ 의미 열을 자동 매핑합니다: "
                + " | ".join(mapped_descriptions),
                "직접 비율은 백분율로 정규화하고, 양 데이터는 trapped+free를 "
                "분모로 비율을 계산합니다.",
                f"행별 계산 근거가 포함된 요약표를 {summary_path}에 저장합니다.",
                f"srCO₂ 또는 시간에 따른 trapped/free 변화 그래프를 {chart_path}에 저장합니다.",
                f"통계·상관·가정이 포함된 보고서를 {report_path}에 저장하고 "
                "세 결과물을 검증합니다.",
            ],
            actions=[*read_actions, run_action],
        )

    def _require_mrst_csv_targets(self, targets: list[str]) -> list[str]:
        if not targets:
            return [self.permissions.to_relative(self.permissions.find_first_file(".csv"))]
        resolved: list[str] = []
        for target in targets:
            path = self.permissions.resolve_path(
                target,
                must_exist=True,
                allow_directory=False,
                allowed_extensions={".csv"},
            )
            relative = self.permissions.to_relative(path)
            if relative not in resolved:
                resolved.append(relative)
        return resolved

    @classmethod
    def _match_mrst_columns(cls, columns: list[str]) -> dict[str, str]:
        normalized_columns = {
            cls._normalize_column_text(column): column for column in columns
        }
        matched: dict[str, str] = {}
        for meaning, aliases in _MRST_COLUMN_ALIASES.items():
            for alias in (meaning, *aliases):
                column = normalized_columns.get(cls._normalize_column_text(alias))
                if column is not None:
                    matched[meaning] = column
                    break
        return matched

    @staticmethod
    def _srco2_from_filename(csv_path: str) -> float | None:
        stem = Path(csv_path).stem.casefold().replace("co₂", "co2")
        match = re.search(r"sr[_-]?co2(?:[_= -])?(\d+(?:\.\d+)?)", stem)
        return float(match.group(1)) if match else None

    def _plan_multi_csv_comparison(
        self,
        request: AgentPlanRequest,
        text: str,
        targets: list[str],
        output: str | None,
    ) -> PlannedTask:
        csv_paths = self._require_csv_targets(targets)
        output_path = output or "results/csv_comparison.csv"
        output_file = self.permissions.resolve_path(
            output_path,
            must_exist=False,
            allow_directory=False,
            allowed_extensions={".csv"},
        )
        output_path = self.permissions.to_relative(output_file)
        chart_path = str(Path(output_path).with_suffix(".png")).replace("\\", "/")
        self.permissions.resolve_path(
            chart_path,
            must_exist=False,
            allow_directory=False,
            allowed_extensions={".png"},
        )

        csv_infos = [self._csv_info(path) for path in csv_paths]
        first_columns = [str(column) for column in csv_infos[0]["columns"]]
        common_columns = [
            column
            for column in first_columns
            if all(column in info["columns"] for info in csv_infos[1:])
        ]
        common_numeric = [
            column
            for column in common_columns
            if all(column in self._numeric_sample_columns(info) for info in csv_infos)
        ]
        if not common_numeric:
            raise ValueError("모든 CSV에서 함께 비교할 수 있는 공통 숫자 열이 없습니다.")
        compare_column = self._select_compare_column(
            request,
            text,
            common_numeric,
        )

        code = self._multi_csv_comparison_code(
            csv_paths,
            output_path,
            chart_path,
            compare_column,
        )
        self.python_tools.validate(code)
        read_actions = [
            self._action(
                AgentToolName.READ_FILE,
                f"비교 CSV 구조 확인: {path}",
                {"path": path, "start_line": 1, "end_line": 25},
                target_files=[path],
            )
            for path in csv_paths
        ]
        run_action = self._action(
            AgentToolName.RUN_PYTHON,
            f"{len(csv_paths)}개 CSV의 {compare_column} 열 비교",
            {
                "code": code,
                "input_paths": csv_paths,
                "common_columns": common_columns,
                "compare_column": compare_column,
                "expected_outputs": [output_path, chart_path],
            },
            target_files=[*csv_paths, output_path, chart_path],
            requires_approval=True,
            preview=code,
        )
        return PlannedTask(
            plan=[
                f"{len(csv_paths)}개 CSV 파일의 열 구조와 데이터 행을 확인합니다.",
                f"공통 열 {', '.join(common_columns)}을 확인합니다.",
                f"공통 숫자 열 {compare_column}의 개수·최소·최대·평균을 파일별로 계산합니다.",
                f"통합 비교표를 {output_path}에 저장합니다.",
                f"파일별 비교 그래프를 {chart_path}에 저장하고 두 결과물을 검증합니다.",
            ],
            actions=[*read_actions, run_action],
        )

    def _select_compare_column(
        self,
        request: AgentPlanRequest,
        text: str,
        common_numeric: list[str],
    ) -> str:
        explicit = (request.compare_column or "").strip()
        if explicit:
            return self._resolve_column_name(explicit, common_numeric)
        mentioned = self._mentioned_columns(text, common_numeric)
        return mentioned[0] if mentioned else common_numeric[0]

    def _plan_csv_chart(
        self,
        request: AgentPlanRequest,
        text: str,
        target: str | None,
        output: str | None,
        chart_type: AgentChartType,
    ) -> PlannedTask:
        csv_path = self._require_csv_target(target)
        output_path = output or _CHART_DEFAULT_OUTPUTS[chart_type]
        self.permissions.resolve_path(
            output_path,
            must_exist=False,
            allow_directory=False,
            allowed_extensions={".png"},
        )
        csv_info = self._csv_info(csv_path)
        label = _CHART_LABELS[chart_type]

        if chart_type == AgentChartType.HISTOGRAM:
            x_name = self._select_histogram_column(request, text, csv_info)
            y_name = None
            column_plan = f"분포를 확인할 숫자 열은 {x_name}으로 확정합니다."
            data_plan = "숫자 변환이 가능한 값을 모아 구간별 빈도를 계산합니다."
            description = f"{x_name} 열의 {label} 생성"
        else:
            x_name, y_name = self._select_two_axis_columns(
                request,
                text,
                csv_info,
                label,
            )
            column_plan = f"X축은 {x_name}, Y축은 {y_name} 열로 확정합니다."
            data_plan = "두 열의 숫자 데이터에서 결측값과 변환 불가능한 행을 제외합니다."
            description = f"{x_name}와 {y_name} 열의 {label} 생성"

        code = self._chart_code(
            csv_path,
            output_path,
            chart_type,
            x_name,
            y_name,
        )
        self.python_tools.validate(code)
        arguments = {
            "code": code,
            "chart_type": chart_type.value,
            "x_column": x_name,
            "expected_outputs": [output_path],
        }
        if y_name is not None:
            arguments["y_column"] = y_name

        return PlannedTask(
            plan=[
                f"{csv_path}의 열 구조와 데이터 개수를 확인합니다.",
                column_plan,
                data_plan,
                f"승인 후 Python으로 {label}를 생성합니다.",
                f"그래프를 {output_path}에 저장하고 PNG 무결성을 검증합니다.",
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
                    description,
                    arguments,
                    target_files=[csv_path, output_path],
                    requires_approval=True,
                    preview=code,
                ),
            ],
        )

    def _select_two_axis_columns(
        self,
        request: AgentPlanRequest,
        text: str,
        csv_info: dict,
        chart_label: str,
    ) -> tuple[str, str]:
        columns = [str(column) for column in csv_info["columns"]]
        explicit_x = (request.x_column or "").strip()
        explicit_y = (request.y_column or "").strip()

        if bool(explicit_x) != bool(explicit_y):
            raise ValueError(f"{chart_label}에는 X축과 Y축 열을 모두 선택해야 합니다.")

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
                f"{chart_label}에 사용할 숫자 열이 2개 이상 필요합니다. "
                f"사용 가능한 열: {', '.join(columns)}"
            )
        return numeric[0], numeric[1]

    def _select_histogram_column(
        self,
        request: AgentPlanRequest,
        text: str,
        csv_info: dict,
    ) -> str:
        columns = [str(column) for column in csv_info["columns"]]
        numeric = self._numeric_sample_columns(csv_info)
        explicit = (request.x_column or request.y_column or "").strip()
        if explicit:
            selected = self._resolve_column_name(explicit, columns)
            if selected not in numeric:
                raise ValueError(
                    f"히스토그램에 사용할 '{selected}' 열에서 숫자 데이터를 "
                    "확인할 수 없습니다."
                )
            return selected

        mentioned = self._mentioned_columns(text, columns)
        for column in mentioned:
            if column in numeric:
                return column
        if numeric:
            return numeric[0]
        raise ValueError(
            "히스토그램에 사용할 숫자 열이 필요합니다. "
            f"사용 가능한 열: {', '.join(columns)}"
        )

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
        lowered = re.sub(r"[^\w가-힣]+", " ", lowered)
        return re.sub(r"[\s_]+", " ", lowered).strip()

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
        paths = AgentPlanner._extract_paths(text)
        return paths[0] if paths else None

    @staticmethod
    def _extract_paths(text: str) -> list[str]:
        allowed = {
            ".txt", ".md", ".py", ".json", ".csv", ".yaml", ".yml", ".log", ".png"
        }
        paths: list[str] = []
        quoted = re.findall(r"[\"'`](.+?)[\"'`]", text)
        for value in quoted:
            candidate = value.strip()
            if Path(candidate).suffix.lower() in allowed and candidate not in paths:
                paths.append(candidate)

        matches = re.findall(
            r"([\w가-힣.()\-\\/]+\.(?:txt|md|py|json|csv|ya?ml|log|png))",
            text,
            flags=re.IGNORECASE,
        )
        for value in matches:
            candidate = value.strip()
            if candidate not in paths:
                paths.append(candidate)
        return paths

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
        if "csv" not in text:
            return False
        statistical_terms = (
            "통계",
            "평균",
            "중앙값",
            "표준편차",
            "상관",
            "pearson",
            "보고서",
            "report",
        )
        if any(word in text for word in statistical_terms):
            return True
        structure_terms = ("구조", "열 이름", "column", "행 수", "데이터 개수")
        return "분석" in text and not any(word in text for word in structure_terms)

    @staticmethod
    def _looks_like_mrst_co2_analysis(
        request: AgentPlanRequest,
        text: str,
        target: str | None,
        targets: list[str],
    ) -> bool:
        if request.analysis_profile == AgentAnalysisProfile.MRST_CO2:
            return True
        has_csv = bool(targets) or bool(target and target.lower().endswith(".csv"))
        if not has_csv:
            return False
        domain_terms = (
            "mrst",
            "co2",
            "co₂",
            "지중 저장",
            "지중저장",
            "trapped_ratio",
            "free_ratio",
            "포획 비율",
            "포획비율",
            "잔류 포화도",
        )
        analysis_terms = (
            "분석",
            "요약",
            "보고서",
            "경향",
            "추세",
            "analyze",
            "analysis",
            "report",
        )
        return any(term in text for term in domain_terms) and any(
            term in text for term in analysis_terms
        )

    @staticmethod
    def _looks_like_multi_csv_compare(
        request: AgentPlanRequest,
        text: str,
        targets: list[str],
    ) -> bool:
        if len(targets) < 2:
            return False
        if request.target_paths:
            return True
        return any(
            word in text
            for word in ("비교", "대조", "차이", "조건별", "compare", "comparison")
        )

    @staticmethod
    def _requested_chart_type(
        request: AgentPlanRequest,
        text: str,
        target: str | None,
    ) -> AgentChartType | None:
        if request.chart_type is not None:
            return request.chart_type

        has_csv = "csv" in text or bool(target and target.lower().endswith(".csv"))
        if not has_csv:
            return None
        if any(word in text for word in ("히스토그램", "histogram", "분포도")):
            return AgentChartType.HISTOGRAM
        if any(word in text for word in ("막대그래프", "막대 그래프", "bar chart", "bar graph")):
            return AgentChartType.BAR
        if any(word in text for word in ("선 그래프", "선그래프", "꺾은선", "line plot", "line chart")):
            return AgentChartType.LINE
        if any(
            word in text
            for word in ("산점도", "scatter", "그래프", "plot", "chart")
        ):
            return AgentChartType.SCATTER
        return None

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

            numeric_columns = {{}}
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
                if values:
                    numeric_columns[column] = values
                lines.append(f"[{{column}}]")
                lines.append(f"- 결측값: {{missing}}")
                if values:
                    lines.append(f"- 숫자 데이터 수: {{len(values)}}")
                    lines.append(f"- 최소: {{min(values):.6g}}")
                    lines.append(f"- 최대: {{max(values):.6g}}")
                    lines.append(f"- 평균: {{statistics.fmean(values):.6g}}")
                    lines.append(f"- 중앙값: {{statistics.median(values):.6g}}")
                    if len(values) >= 2:
                        lines.append(
                            f"- 표준편차(표본): {{statistics.stdev(values):.6g}}"
                        )
                    else:
                        lines.append("- 표준편차(표본): 계산에 2개 이상의 값이 필요함")
                else:
                    lines.append("- 숫자 열이 아님")
                lines.append("")

            lines.append("[Pearson 상관계수]")
            numeric_names = list(numeric_columns)
            if len(numeric_names) < 2:
                lines.append("- 숫자 열이 2개 이상 필요함")
            for left_index in range(len(numeric_names)):
                for right_index in range(left_index + 1, len(numeric_names)):
                    left = numeric_names[left_index]
                    right = numeric_names[right_index]
                    left_values = []
                    right_values = []
                    for row in rows:
                        try:
                            left_value = float((row.get(left) or "").strip())
                            right_value = float((row.get(right) or "").strip())
                        except ValueError:
                            continue
                        left_values.append(left_value)
                        right_values.append(right_value)
                    if len(left_values) < 2:
                        value_text = "계산에 2개 이상의 짝지어진 값이 필요함"
                    elif len(set(left_values)) < 2 or len(set(right_values)) < 2:
                        value_text = "한 열의 분산이 0이어서 계산할 수 없음"
                    else:
                        correlation = statistics.correlation(left_values, right_values)
                        value_text = f"{{correlation:.6g}} (n={{len(left_values)}})"
                    lines.append(f"- {{left}} ↔ {{right}}: {{value_text}}")

            with open(output_path, "w", encoding="utf-8") as handle:
                handle.write(chr(10).join(lines))

            print(f"Saved report: {{output_path}}")
            """
        ).strip()

    @staticmethod
    def _mrst_co2_analysis_code(
        csv_paths: list[str],
        column_maps: dict[str, dict[str, str]],
        condition_fallbacks: dict[str, float | None],
        summary_path: str,
        chart_path: str,
        report_path: str,
    ) -> str:
        return textwrap.dedent(
            f"""
            import csv
            import statistics
            import matplotlib.pyplot as plt

            input_paths = {csv_paths!r}
            column_maps = {column_maps!r}
            condition_fallbacks = {condition_fallbacks!r}
            summary_path = {summary_path!r}
            chart_path = {chart_path!r}
            report_path = {report_path!r}

            def number(row, column):
                if not column:
                    return None
                raw = (row.get(column) or "").strip()
                if not raw:
                    return None
                try:
                    return float(raw)
                except ValueError:
                    return None

            def percent(value):
                if value is None:
                    return None
                return value * 100.0 if abs(value) <= 1.0 else value

            def formatted(value):
                return "" if value is None else f"{{value:.12g}}"

            def storage_unit(column):
                lowered = (column or "").casefold().replace(" ", "_")
                if "_mt" in lowered or "megaton" in lowered:
                    return "Mt"
                if "_kg" in lowered:
                    return "kg"
                if "tonne" in lowered or "_ton" in lowered:
                    return "tonne"
                if "volume" in lowered:
                    return "input_volume_unit"
                return "input_unit_unspecified"

            def correlation(rows, x_key, y_key):
                pairs = [
                    (row[x_key], row[y_key])
                    for row in rows
                    if row[x_key] is not None and row[y_key] is not None
                ]
                if len(pairs) < 2:
                    return None, len(pairs), "짝지어진 값이 2개 미만"
                x_values = [pair[0] for pair in pairs]
                y_values = [pair[1] for pair in pairs]
                if len(set(x_values)) < 2 or len(set(y_values)) < 2:
                    return None, len(pairs), "한 변수의 분산이 0"
                return statistics.correlation(x_values, y_values), len(pairs), ""

            result_rows = []
            skipped_rows = []
            ratio_sum_mismatch = 0

            for input_path in input_paths:
                mapping = column_maps[input_path]
                with open(input_path, "r", encoding="utf-8-sig", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                for row_index, row in enumerate(rows, start=1):
                    trapped_amount = number(row, mapping.get("trapped_amount"))
                    free_amount = number(row, mapping.get("free_amount"))
                    trapped_ratio = percent(number(row, mapping.get("trapped_ratio")))
                    free_ratio = percent(number(row, mapping.get("free_ratio")))
                    basis = "direct_ratio"

                    if trapped_ratio is None and free_ratio is not None:
                        trapped_ratio = 100.0 - free_ratio
                        basis = "direct_free_ratio_plus_complement"
                    elif free_ratio is None and trapped_ratio is not None:
                        free_ratio = 100.0 - trapped_ratio
                        basis = "direct_trapped_ratio_plus_complement"
                    elif trapped_ratio is None and free_ratio is None:
                        if trapped_amount is None or free_amount is None:
                            skipped_rows.append(f"{{input_path}} row {{row_index}}: ratio/amount missing")
                            continue
                        denominator = trapped_amount + free_amount
                        if denominator == 0:
                            skipped_rows.append(f"{{input_path}} row {{row_index}}: trapped+free is zero")
                            continue
                        trapped_ratio = trapped_amount / denominator * 100.0
                        free_ratio = free_amount / denominator * 100.0
                        basis = "derived_from_trapped_and_free_amounts"

                    if abs((trapped_ratio + free_ratio) - 100.0) > 0.5:
                        ratio_sum_mismatch += 1

                    srco2 = number(row, mapping.get("srco2"))
                    if srco2 is None:
                        srco2 = condition_fallbacks.get(input_path)
                    time_years = number(row, mapping.get("time"))
                    total_storage = number(row, mapping.get("total_storage"))
                    if total_storage is None and trapped_amount is not None and free_amount is not None:
                        total_storage = trapped_amount + free_amount
                    unit_column = (
                        mapping.get("total_storage")
                        or mapping.get("trapped_amount")
                        or mapping.get("free_amount")
                    )
                    result_rows.append(
                        {{
                            "source_file": input_path,
                            "row_index": row_index,
                            "srco2": srco2,
                            "time_years": time_years,
                            "trapped_ratio_pct": trapped_ratio,
                            "free_ratio_pct": free_ratio,
                            "trapped_amount": trapped_amount,
                            "free_amount": free_amount,
                            "total_storage_value": total_storage,
                            "storage_unit": storage_unit(unit_column),
                            "calculation_basis": basis,
                        }}
                    )

            if not result_rows:
                raise ValueError("분석할 수 있는 MRST/CO2 데이터 행이 없습니다.")

            fieldnames = [
                "source_file",
                "row_index",
                "srco2",
                "time_years",
                "trapped_ratio_pct",
                "free_ratio_pct",
                "trapped_amount",
                "free_amount",
                "total_storage_value",
                "storage_unit",
                "calculation_basis",
            ]
            with open(summary_path, "w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for result in result_rows:
                    writer.writerow(
                        {{
                            key: formatted(result[key])
                            if key in {{
                                "srco2",
                                "time_years",
                                "trapped_ratio_pct",
                                "free_ratio_pct",
                                "trapped_amount",
                                "free_amount",
                                "total_storage_value",
                            }}
                            else result[key]
                            for key in fieldnames
                        }}
                    )

            srco2_points = [row for row in result_rows if row["srco2"] is not None]
            time_points = [row for row in result_rows if row["time_years"] is not None]
            plt.figure(figsize=(9, 5.5))
            if len(srco2_points) >= 2 and len(set(row["srco2"] for row in srco2_points)) >= 2:
                plotted = sorted(srco2_points, key=lambda row: row["srco2"])
                x_values = [row["srco2"] for row in plotted]
                x_label = "Residual CO2 saturation (srCO2)"
            elif len(time_points) >= 2 and len(set(row["time_years"] for row in time_points)) >= 2:
                plotted = sorted(time_points, key=lambda row: row["time_years"])
                x_values = [row["time_years"] for row in plotted]
                x_label = "Simulation time (years)"
            else:
                plotted = result_rows
                x_values = list(range(1, len(plotted) + 1))
                x_label = "Analyzed row"
            plt.plot(
                x_values,
                [row["trapped_ratio_pct"] for row in plotted],
                marker="o",
                label="Trapped CO2",
            )
            plt.plot(
                x_values,
                [row["free_ratio_pct"] for row in plotted],
                marker="o",
                label="Free/mobile CO2",
            )
            plt.xlabel(x_label)
            plt.ylabel("CO2 ratio (%)")
            plt.title("MRST CO2 storage analysis")
            plt.ylim(bottom=0)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            with open(chart_path, "wb"):
                pass
            plt.savefig(chart_path, dpi=160)
            plt.close()

            trapped_values = [row["trapped_ratio_pct"] for row in result_rows]
            free_values = [row["free_ratio_pct"] for row in result_rows]
            storage_values = [
                row["total_storage_value"]
                for row in result_rows
                if row["total_storage_value"] is not None
            ]
            lines = [
                "# MRST / CO2 Storage Analysis",
                "",
                "## Inputs",
                f"- Source CSV files: {{len(input_paths)}}",
                f"- Analyzed rows: {{len(result_rows)}}",
                f"- Skipped rows: {{len(skipped_rows)}}",
            ]
            for input_path in input_paths:
                mapping_text = ", ".join(
                    f"{{meaning}}={{column}}"
                    for meaning, column in column_maps[input_path].items()
                )
                lines.append(f"- `{{input_path}}`: {{mapping_text}}")

            lines.extend(
                [
                    "",
                    "## Ratio summary",
                    f"- Trapped CO2 (%): mean {{statistics.fmean(trapped_values):.6g}}, "
                    f"min {{min(trapped_values):.6g}}, max {{max(trapped_values):.6g}}",
                    f"- Free/mobile CO2 (%): mean {{statistics.fmean(free_values):.6g}}, "
                    f"min {{min(free_values):.6g}}, max {{max(free_values):.6g}}",
                    f"- Rows where trapped + free differs from 100% by more than 0.5%p: "
                    f"{{ratio_sum_mismatch}}",
                ]
            )
            if storage_values:
                units = sorted(set(row["storage_unit"] for row in result_rows if row["total_storage_value"] is not None))
                lines.extend(
                    [
                        "",
                        "## Storage summary",
                        f"- Input units: {{', '.join(units)}}",
                        f"- Total storage input value: mean {{statistics.fmean(storage_values):.6g}}, "
                        f"min {{min(storage_values):.6g}}, max {{max(storage_values):.6g}}",
                    ]
                )

            lines.extend(["", "## Correlation and trend"])
            for x_key, x_label_text in (("srco2", "srCO2"), ("time_years", "time_years")):
                for y_key, y_label_text in (("trapped_ratio_pct", "trapped ratio"), ("free_ratio_pct", "free ratio")):
                    value, count, reason = correlation(result_rows, x_key, y_key)
                    if value is None:
                        lines.append(f"- {{x_label_text}} vs {{y_label_text}}: not calculated ({{reason}}, n={{count}})")
                    else:
                        direction = "positive" if value > 0 else "negative" if value < 0 else "none"
                        lines.append(f"- {{x_label_text}} vs {{y_label_text}}: Pearson r={{value:.6g}} (n={{count}}, {{direction}} correlation)")

            lines.extend(
                [
                    "",
                    "## Calculation rules and assumptions",
                    "- Original CSV files were read only and were not modified.",
                    "- Direct ratio values with absolute value <= 1 were interpreted as fractions and multiplied by 100.",
                    "- If only one direct ratio was present, the other was calculated as 100 minus that ratio.",
                    "- Amount-based ratios used trapped / (trapped + free) and free / (trapped + free).",
                    "- `storage_unit=input_unit_unspecified` means the source column did not encode a recognized unit.",
                    "- Correlation describes only the supplied input data and does not establish causality.",
                    "",
                    "## Outputs",
                    f"- Processed summary: `{{summary_path}}`",
                    f"- Ratio chart: `{{chart_path}}`",
                ]
            )
            if skipped_rows:
                lines.extend(["", "## Skipped rows", *[f"- {{item}}" for item in skipped_rows]])

            with open(report_path, "w", encoding="utf-8") as handle:
                handle.write(chr(10).join(lines))

            print(f"Saved MRST summary: {{summary_path}}")
            print(f"Saved MRST chart: {{chart_path}}")
            print(f"Saved MRST report: {{report_path}}")
            """
        ).strip()

    @staticmethod
    def _multi_csv_comparison_code(
        csv_paths: list[str],
        output_path: str,
        chart_path: str,
        compare_column: str,
    ) -> str:
        return textwrap.dedent(
            f"""
            import csv
            import statistics
            import matplotlib.pyplot as plt

            input_paths = {csv_paths!r}
            output_path = {output_path!r}
            chart_path = {chart_path!r}
            compare_column = {compare_column!r}
            labels = {[Path(path).stem for path in csv_paths]!r}
            summary_rows = []

            for input_path in input_paths:
                with open(input_path, "r", encoding="utf-8-sig", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                if not rows:
                    raise ValueError(f"CSV에 데이터 행이 없습니다: {{input_path}}")
                if compare_column not in rows[0]:
                    raise ValueError(
                        f"CSV에 {{compare_column}} 열이 없습니다: {{input_path}}"
                    )

                values = []
                for row in rows:
                    raw = (row.get(compare_column) or "").strip()
                    if not raw:
                        continue
                    try:
                        values.append(float(raw))
                    except ValueError:
                        continue
                if not values:
                    raise ValueError(
                        f"{{input_path}}의 {{compare_column}} 열에 숫자 데이터가 없습니다."
                    )
                summary_rows.append(
                    {{
                        "source_file": input_path,
                        "comparison_column": compare_column,
                        "count": len(values),
                        "minimum": min(values),
                        "maximum": max(values),
                        "mean": statistics.fmean(values),
                    }}
                )

            with open(output_path, "w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "source_file",
                        "comparison_column",
                        "count",
                        "minimum",
                        "maximum",
                        "mean",
                    ],
                )
                writer.writeheader()
                writer.writerows(summary_rows)

            positions = list(range(len(summary_rows)))
            width = 0.25
            plt.figure(figsize=(max(8, len(summary_rows) * 1.4), 5))
            plt.bar(
                [position - width for position in positions],
                [row["minimum"] for row in summary_rows],
                width=width,
                label="Minimum",
            )
            plt.bar(
                positions,
                [row["mean"] for row in summary_rows],
                width=width,
                label="Mean",
            )
            plt.bar(
                [position + width for position in positions],
                [row["maximum"] for row in summary_rows],
                width=width,
                label="Maximum",
            )
            plt.xticks(positions, labels, rotation=20, ha="right")
            plt.ylabel(compare_column)
            plt.title(f"CSV comparison: {{compare_column}}")
            plt.legend()
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            with open(chart_path, "wb"):
                pass
            plt.savefig(chart_path, dpi=160)
            plt.close()
            print(f"Saved comparison table: {{output_path}}")
            print(f"Saved comparison chart: {{chart_path}}")
            """
        ).strip()

    @staticmethod
    def _chart_code(
        csv_path: str,
        output_path: str,
        chart_type: AgentChartType,
        x_name: str,
        y_name: str | None,
    ) -> str:
        return textwrap.dedent(
            f"""
            import csv
            import matplotlib.pyplot as plt

            input_path = {csv_path!r}
            output_path = {output_path!r}
            chart_type = {chart_type.value!r}
            x_name = {x_name!r}
            y_name = {y_name!r}

            with open(input_path, "r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))

            if not rows:
                raise ValueError("CSV에 데이터 행이 없습니다.")

            columns = list(rows[0].keys())
            required_columns = [x_name] if y_name is None else [x_name, y_name]
            missing_columns = [name for name in required_columns if name not in columns]
            if missing_columns:
                raise ValueError("CSV 열을 찾을 수 없습니다: " + ", ".join(missing_columns))

            x_values = []
            y_values = []
            for row in rows:
                try:
                    x_value = float((row.get(x_name) or "").strip())
                    if y_name is not None:
                        y_value = float((row.get(y_name) or "").strip())
                except ValueError:
                    continue
                x_values.append(x_value)
                if y_name is not None:
                    y_values.append(y_value)

            if not x_values:
                raise ValueError("선택한 열에 사용할 수 있는 숫자 행이 없습니다.")

            plt.figure(figsize=(8, 5))
            if chart_type == "histogram":
                bins = min(20, max(5, round(len(x_values) ** 0.5)))
                plt.hist(x_values, bins=bins, edgecolor="black", alpha=0.8)
                plt.xlabel(x_name)
                plt.ylabel("Frequency")
                plt.title(f"Distribution of {{x_name}}")
            else:
                if chart_type == "line":
                    pairs = sorted(zip(x_values, y_values))
                    x_values = [pair[0] for pair in pairs]
                    y_values = [pair[1] for pair in pairs]
                    plt.plot(x_values, y_values, marker="o")
                elif chart_type == "bar":
                    unique_x = sorted(set(x_values))
                    gaps = [
                        right - left
                        for left, right in zip(unique_x, unique_x[1:])
                        if right > left
                    ]
                    bar_width = min(gaps) * 0.7 if gaps else 0.8
                    plt.bar(x_values, y_values, width=bar_width)
                else:
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
            print(f"Saved {{chart_type}} chart: {{output_path}}")
            """
        ).strip()
