"""Markdown parser for Python code blocks with attributes."""

from __future__ import annotations
import hashlib
import re
import yaml
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple


@dataclass
class DocumentConfig:
    """Document configuration from frontmatter."""

    title: str = ""
    author: str = ""
    theme: str = "dark"  # color scheme: "dark", "light", or "auto"
    ui_theme: Optional[str] = None  # css theme: "default" or "none"; None = unspecified
    show_widgets: Optional[bool] = None  # widgets/menu visibility; None = default (on)
    syntax_theme: str = "monokai"  # pygments style name
    show_line_numbers: bool = False
    collapse_code: bool = True
    custom_css: str = ""
    code_font_size: Optional[Any] = None  # CSS size token or number (px)


@dataclass
class CodeCell:
    """Represents a Python code cell with metadata."""

    id: str
    code: str
    deps: List[str]
    outputs: List[str]
    needs: List[str]  # Local cell dependencies within same file
    needs_files: List[str]  # Cross-file dependencies (e.g., "../math/algebra.md:cellX")
    line_start: int
    line_end: int
    collapse_code: bool = False
    collapse_output: bool = False
    commented: bool = False
    timeout: Optional[int] = None  # Custom timeout in seconds


def parse_attributes(info_string: str) -> Dict[str, str]:
    """Parse attributes from code fence info string.

    Example: python id=cell1 deps=numpy,pandas outputs=plot.png
    Returns: {'id': 'cell1', 'deps': 'numpy,pandas', 'outputs': 'plot.png'}
    """
    attrs = {}
    if not info_string.strip():
        return attrs

    parts = info_string.split()
    if not parts or (parts[0] != "python" and parts[0] != "```python"):
        return attrs

    # Skip the first part (either 'python' or '```python')
    start_idx = 1

    for part in parts[start_idx:]:
        if "=" in part:
            key, value = part.split("=", 1)
            attrs[key] = value

    return attrs


def parse_cell_needs(needs_attr: str) -> Tuple[List[str], List[str]]:
    """
    Parse needs attribute into local cell dependencies and file dependencies.

    Local dependencies are simple cell IDs within the same file.
    File dependencies are paths with optional cell ID:
      - "../math/algebra.md" - depend on entire file
      - "../math/algebra.md:cellX" - depend on specific cell in file
      - "math/algebra.md" - relative path
      - "/absolute/path/file.md:cellId" - absolute path

    Args:
        needs_attr: Comma-separated list of dependencies

    Returns:
        Tuple of (local_cell_ids, file_references)

    Examples:
        "cellA,cellB" -> (["cellA", "cellB"], [])
        "../math/algebra.md:cellX" -> ([], ["../math/algebra.md:cellX"])
        "cellA,../math/algebra.md" -> (["cellA"], ["../math/algebra.md"])
        "cellA,../math/algebra.md:cellX,cellB" -> (["cellA", "cellB"], ["../math/algebra.md:cellX"])
    """
    if not needs_attr:
        return [], []

    local = []
    files = []

    for need in needs_attr.split(','):
        need = need.strip()
        if not need:
            continue

        # Check if this looks like a file path
        # File paths contain '/' or end with '.md'
        if '/' in need or need.endswith('.md'):
            files.append(need)
        else:
            local.append(need)

    return local, files


def generate_cell_id(code: str) -> str:
    """Generate a unique ID for a code cell based on its content."""
    return hashlib.sha256(code.encode()).hexdigest()[:8]


def parse_frontmatter(content: str) -> Tuple[DocumentConfig, str]:
    """Parse YAML frontmatter and return config + remaining content."""
    lines = content.splitlines()

    # Check if content starts with frontmatter
    if not lines or lines[0].strip() != "---":
        # No frontmatter - use defaults but don't collapse code
        return DocumentConfig(collapse_code=False), content

    # Find the closing ---
    frontmatter_end = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            frontmatter_end = i
            break

    if frontmatter_end == -1:
        # Incomplete frontmatter - treat as no frontmatter
        return DocumentConfig(collapse_code=False), content

    # Parse YAML frontmatter
    try:
        frontmatter_content = "\n".join(lines[1:frontmatter_end])
        if not frontmatter_content.strip():
            frontmatter_data = {}
        else:
            frontmatter_data = yaml.safe_load(frontmatter_content) or {}
    except yaml.YAMLError:
        frontmatter_data = {}

    # Create config with defaults
    # Support both 'widgets' and 'show_widgets' keys
    _widgets_value = frontmatter_data.get("widgets")
    if _widgets_value is None:
        _widgets_value = frontmatter_data.get("show_widgets")

    config = DocumentConfig(
        title=frontmatter_data.get("title", ""),
        author=frontmatter_data.get("author", ""),
        theme=frontmatter_data.get("theme", "dark"),
        # If not specified in frontmatter, leave None so template can decide precedence
        ui_theme=frontmatter_data.get("ui_theme"),
        show_widgets=_widgets_value,
        syntax_theme=frontmatter_data.get("syntax_theme", "monokai"),
        show_line_numbers=frontmatter_data.get("show_line_numbers", False),
        collapse_code=frontmatter_data.get("collapse_code", False),
        custom_css=frontmatter_data.get("custom_css", ""),
        code_font_size=frontmatter_data.get("code_font_size"),
    )

    # Return remaining content
    remaining_content = "\n".join(lines[frontmatter_end + 1 :])
    return config, remaining_content


def parse_markdown(content: str) -> Tuple[DocumentConfig, List[CodeCell]]:
    """Parse markdown content and extract config + Python code cells."""
    config, markdown_content = parse_frontmatter(content)

    cells = []
    lines = markdown_content.splitlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Look for Python code fence (normal or commented)
        commented = False
        python_fence_line = None

        if line.startswith("```python"):
            python_fence_line = line
        elif line.startswith("<!-- ```python") and line.endswith("-->"):
            # HTML comment style: <!-- ```python -->
            python_fence_line = line[4:-3].strip()  # Remove <!-- and -->
            commented = True
        elif line.startswith("#") and "```python" in line:
            # Markdown comment style: # ```python
            python_fence_line = line.lstrip("#").strip()
            commented = True

        if python_fence_line:
            line_start = i + 1
            attrs = parse_attributes(python_fence_line)  # Full line for parsing

            # Find the end of the code block
            code_lines = []
            i += 1
            while i < len(lines):
                current_line = lines[i].strip()
                # Check for end fence (normal or commented)
                if current_line.startswith("```"):
                    break
                elif commented and (
                    current_line.startswith("<!-- ```")
                    or current_line.startswith("# ```")
                ):
                    break
                else:
                    code_lines.append(lines[i])
                    i += 1

            if code_lines:
                code = "\n".join(code_lines)
                cell_id = attrs.get("id", generate_cell_id(code))

                # Parse dependencies
                deps = []
                if "deps" in attrs:
                    deps = [dep.strip() for dep in attrs["deps"].split(",")]

                # Parse outputs
                outputs = []
                if "outputs" in attrs:
                    outputs = [out.strip() for out in attrs["outputs"].split(",")]

                # Parse needs/depends (dependencies on other cells and files)
                # Support both 'needs' and 'depends' as aliases
                dep_attr = None
                if "needs" in attrs:
                    dep_attr = attrs["needs"]
                elif "depends" in attrs:
                    dep_attr = attrs["depends"]

                # Separate local cell dependencies from file dependencies
                needs, needs_files = parse_cell_needs(dep_attr or "")

                # Parse collapse attributes
                collapse_code = attrs.get("collapse-code", "").lower() == "true"
                collapse_output = attrs.get("collapse-output", "").lower() == "true"

                # Parse collapsed attribute (convenience parameter that sets both)
                if "collapsed" in attrs:
                    collapsed_value = attrs.get("collapsed", "").lower() == "true"
                    collapse_code = collapsed_value
                    collapse_output = collapsed_value

                # Parse timeout attribute (in seconds)
                timeout = None
                if "timeout" in attrs:
                    try:
                        timeout = int(attrs["timeout"])
                    except (ValueError, TypeError):
                        # Invalid timeout value, use default
                        pass

                cell = CodeCell(
                    id=cell_id,
                    code=code,
                    deps=deps,
                    outputs=outputs,
                    needs=needs,
                    needs_files=needs_files,
                    line_start=line_start,
                    line_end=i,
                    collapse_code=collapse_code,
                    collapse_output=collapse_output,
                    commented=commented,
                    timeout=timeout,
                )
                cells.append(cell)

        i += 1

    return config, cells


def validate_cells(cells: List[CodeCell]) -> None:
    """Validate that cell dependencies are satisfied."""
    cell_ids = {cell.id for cell in cells}

    # Duplicate ID detection
    seen: Dict[str, List[CodeCell]] = {}
    for c in cells:
        seen.setdefault(c.id, []).append(c)
    dupes = {cid: lst for cid, lst in seen.items() if len(lst) > 1}
    if dupes:
        details = ", ".join(
            f"{cid} @ " + ",".join(str(c.line_start) for c in lst)
            for cid, lst in dupes.items()
        )
        raise ValueError(f"Duplicate cell ids: {details}")

    # Unknown dependency detection with line context
    for cell in cells:
        for need in cell.needs:
            if need not in cell_ids:
                raise ValueError(
                    f"Cell '{cell.id}' (line {cell.line_start}) depends on unknown cell '{need}'"
                )
