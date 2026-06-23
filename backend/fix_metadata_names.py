import json
import re

from app.core.config import Settings


settings = Settings()
metadata_dir = settings.metadata_dir

# 파일명 앞의 64자리 SHA-256 해시와 밑줄 제거
hash_prefix = re.compile(r"^[0-9a-fA-F]{64}_")

print(f"메타데이터 폴더: {metadata_dir}")

if not metadata_dir.exists():
    raise FileNotFoundError(
        f"메타데이터 폴더를 찾을 수 없습니다: {metadata_dir}"
    )

changed_count = 0

for path in metadata_dir.glob("*.json"):
    try:
        data = json.loads(
            path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[건너뜀] {path.name}: {exc}")
        continue

    changed = False

    filename = data.get("filename")

    if isinstance(filename, str):
        cleaned_filename = hash_prefix.sub("", filename)

        if cleaned_filename != filename:
            data["filename"] = cleaned_filename
            changed = True

    title = data.get("title")

    if isinstance(title, str):
        cleaned_title = hash_prefix.sub("", title)

        # 제목에 확장자가 그대로 들어 있으면 제거
        for extension in (
            ".pdf",
            ".txt",
            ".ppt",
            ".pptx",
            ".png",
            ".jpg",
            ".jpeg",
        ):
            if cleaned_title.lower().endswith(extension):
                cleaned_title = cleaned_title[:-len(extension)]
                break

        if cleaned_title != title:
            data["title"] = cleaned_title
            changed = True

    if changed:
        path.write_text(
            json.dumps(
                data,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        changed_count += 1
        print(
            f"[수정 완료] {path.name} → "
            f"{data.get('filename', '')}"
        )

print(f"총 {changed_count}개 메타데이터 수정 완료")