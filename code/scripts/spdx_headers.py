#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
from pathlib import Path
from typing import Iterable, Optional

HASH_EXTENSIONS = {
    ".py",
    ".sh",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".txt",
}

HASH_FILENAMES = {
    ".gitignore",
    ".zshrc",
}

C_STYLE_EXTENSIONS = {
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".css",
    ".scss",
    ".java",
}

EXCLUDE_DIRS = {
    ".git",
    ".pytest_cache",
    "frames_data",
    "docker",
    ".tox",
}

EXCLUDE_FILES = {
    Path("code/tests/orientations_outputs.txt"),
}

# Old proprietary license bodies (year-agnostic, used for replacement)
_OLD_HASH_BODY = (
    "# SPDX-License-Identifier: Apache-2.0\n"
    "#\n"
    "# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual\n"
    "# property and proprietary rights in and to this material, related\n"
    "# documentation and any modifications thereto. Any use, reproduction,\n"
    "# disclosure or distribution of this material and related documentation\n"
    "# without an express license agreement from NVIDIA CORPORATION or\n"
    "# its affiliates is strictly prohibited."
)

_OLD_C_STYLE_BODY = (
    " * SPDX-License-Identifier: LicenseRef-NvidiaProprietary\n"
    " *\n"
    " * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual\n"
    " * property and proprietary rights in and to this material, related\n"
    " * documentation and any modifications thereto. Any use, reproduction,\n"
    " * disclosure or distribution of this material and related documentation\n"
    " * without an express license agreement from NVIDIA CORPORATION or\n"
    " * its affiliates is strictly prohibited.\n"
    " */"
)


def _hash_header(year: int) -> str:
    return "\n".join(
        [
            f"# SPDX-FileCopyrightText: Copyright (c) {year} NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
            "# SPDX-License-Identifier: Apache-2.0",
            "#",
            "# Licensed under the Apache License, Version 2.0 (the \"License\");",
            "# you may not use this file except in compliance with the License.",
            "# You may obtain a copy of the License at",
            "#",
            "# http://www.apache.org/licenses/LICENSE-2.0",
            "#",
            "# Unless required by applicable law or agreed to in writing, software",
            "# distributed under the License is distributed on an \"AS IS\" BASIS,",
            "# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.",
            "# See the License for the specific language governing permissions and",
            "# limitations under the License.",
        ]
    )


def _c_style_header(year: int) -> str:
    return "\n".join(
        [
            "/*",
            f" * SPDX-FileCopyrightText: Copyright (c) {year} NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
            " * SPDX-License-Identifier: Apache-2.0",
            " *",
            " * Licensed under the Apache License, Version 2.0 (the \"License\");",
            " * you may not use this file except in compliance with the License.",
            " * You may obtain a copy of the License at",
            " *",
            " * http://www.apache.org/licenses/LICENSE-2.0",
            " *",
            " * Unless required by applicable law or agreed to in writing, software",
            " * distributed under the License is distributed on an \"AS IS\" BASIS,",
            " * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.",
            " * See the License for the specific language governing permissions and",
            " * limitations under the License.",
            " */",
        ]
    )


def _new_hash_body() -> str:
    return (
        "# SPDX-License-Identifier: Apache-2.0\n"
        "#\n"
        "# Licensed under the Apache License, Version 2.0 (the \"License\");\n"
        "# you may not use this file except in compliance with the License.\n"
        "# You may obtain a copy of the License at\n"
        "#\n"
        "# http://www.apache.org/licenses/LICENSE-2.0\n"
        "#\n"
        "# Unless required by applicable law or agreed to in writing, software\n"
        "# distributed under the License is distributed on an \"AS IS\" BASIS,\n"
        "# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n"
        "# See the License for the specific language governing permissions and\n"
        "# limitations under the License."
    )


def _new_c_style_body() -> str:
    return (
        " * SPDX-License-Identifier: Apache-2.0\n"
        " *\n"
        " * Licensed under the Apache License, Version 2.0 (the \"License\");\n"
        " * you may not use this file except in compliance with the License.\n"
        " * You may obtain a copy of the License at\n"
        " *\n"
        " * http://www.apache.org/licenses/LICENSE-2.0\n"
        " *\n"
        " * Unless required by applicable law or agreed to in writing, software\n"
        " * distributed under the License is distributed on an \"AS IS\" BASIS,\n"
        " * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n"
        " * See the License for the specific language governing permissions and\n"
        " * limitations under the License.\n"
        " */"
    )


def _comment_style(path: Path) -> Optional[str]:
    if path.name in HASH_FILENAMES or path.suffix in HASH_EXTENSIONS:
        return "hash"
    if path.suffix in C_STYLE_EXTENSIONS:
        return "c"
    return None


def _should_exclude(path: Path) -> bool:
    for part in path.parts:
        if part in EXCLUDE_DIRS:
            return True
    return path in EXCLUDE_FILES


def _iter_files(root: Path) -> Iterable[Path]:
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=root,
            check=True,
            text=True,
            capture_output=True,
        )
        for line in result.stdout.splitlines():
            path = (root / line).resolve()
            relative = path.relative_to(root)
            if _should_exclude(relative):
                continue
            if _comment_style(path) is None:
                continue
            yield path
        return
    except Exception:
        pass

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in EXCLUDE_DIRS and not d.startswith(".venv") and d != "venv"
        ]
        for filename in filenames:
            path = Path(dirpath) / filename
            if _should_exclude(path.relative_to(root)):
                continue
            if _comment_style(path) is None:
                continue
            yield path


def _has_header(content: str) -> bool:
    prefix = "\n".join(content.splitlines()[:20])
    return "SPDX-FileCopyrightText" in prefix


def _apply_header(path: Path, header: str, style: str) -> bool:
    raw = path.read_text(encoding="utf-8")

    # Replace old proprietary license body with new Apache body (preserves copyright year)
    old_body = _OLD_HASH_BODY if style == "hash" else _OLD_C_STYLE_BODY
    new_body = _new_hash_body() if style == "hash" else _new_c_style_body()
    if old_body in raw:
        new_content = raw.replace(old_body, new_body, 1)
        path.write_text(new_content, encoding="utf-8")
        return True

    if _has_header(raw):
        return False

    lines = raw.splitlines()
    new_lines = []
    if lines and lines[0].startswith("#!"):
        new_lines.append(lines[0])
        new_lines.append(header)
        new_lines.append("")
        new_lines.extend(lines[1:])
    else:
        new_lines.append(header)
        new_lines.append("")
        new_lines.extend(lines)

    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Add or check SPDX headers.")
    parser.add_argument(
        "--check", action="store_true", help="Fail if any files are missing headers."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root.",
    )
    args = parser.parse_args()

    year = dt.datetime.now().year
    missing = []
    updated = []

    for path in sorted(_iter_files(args.root)):
        style = _comment_style(path)
        if style is None:
            continue
        header = _hash_header(year) if style == "hash" else _c_style_header(year)
        if args.check:
            raw = path.read_text(encoding="utf-8")
            if not _has_header(raw):
                missing.append(path.relative_to(args.root))
        else:
            if _apply_header(path, header, style):
                updated.append(path.relative_to(args.root))

    if args.check:
        if missing:
            missing_str = "\n".join(f"- {path}" for path in missing)
            print("Missing SPDX headers:\n" + missing_str)
            return 1
        print("All checked files have SPDX headers.")
        return 0

    if updated:
        updated_str = "\n".join(f"- {path}" for path in updated)
        print("Updated SPDX headers:\n" + updated_str)
    else:
        print("No files updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
