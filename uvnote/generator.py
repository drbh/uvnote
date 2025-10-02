"""Static HTML output generator."""

from __future__ import annotations
import html
import os
import platform
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import markdown
from jinja2 import Environment, BaseLoader, FileSystemLoader, TemplateNotFound
from pygments import highlight
from pygments.formatters.html import HtmlFormatter
from pygments.lexers.python import PythonLexer

from .executor import ExecutionResult
from .parser import CodeCell, DocumentConfig


def get_system_info() -> Dict[str, str]:
    """Collect basic system information for display in generated pages."""
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor() or "Unknown",
        "platform": platform.platform(),
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{{ title }}</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 16px; }
    .system-info { font-size: 12px; color: #666; margin-bottom: 12px; }
  </style>
</head>
<body>
  <div class="system-info">
    {{ system_info.system }} {{ system_info.machine }} | {{ system_info.platform }}
  </div>
  <div class="main-content">
    {{ content | safe }}
  </div>
</body>
</html>"""


def highlight_code(code: str, config: DocumentConfig) -> str:
    """Highlight Python code using Pygments."""
    lexer = PythonLexer()
    formatter = HtmlFormatter(
        style=config.syntax_theme,
        nowrap=False,
        linenos=False,
        linenos_special=1,
        cssclass="highlight",
    )
    return highlight(code, lexer, formatter)


def split_uv_install_logs(stderr: str) -> Tuple[str, str]:
    """Split stderr into UV install logs and regular stderr.

    Returns:
        (uv_install_logs, regular_stderr)
    """
    lines = stderr.split('\n')
    uv_logs = []
    regular_logs = []
    in_uv_section = True

    for line in lines:
        if in_uv_section:
            uv_logs.append(line)
            # Check if we've reached the end of UV install logs
            if line.startswith('Installed '):
                in_uv_section = False
        else:
            regular_logs.append(line)

    # If we never found "Installed", treat it all as regular stderr
    if in_uv_section:
        return "", stderr

    return '\n'.join(uv_logs), '\n'.join(regular_logs).strip()


def render_artifact_preview(artifact: str, cell_id: str, cache_dir: Path, cache_key: str) -> str:
    """Render a single artifact preview as HTML."""
    html_parts = []

    if artifact.endswith((".png", ".jpg", ".jpeg")):
        html_parts.append('<div class="artifact-preview">')
        html_parts.append(
            f'<img src="artifacts/{cell_id}/{artifact}" alt="{artifact}">'
        )
        html_parts.append("</div>")
    elif artifact.endswith(".svg"):
        # Read and embed SVG content directly
        svg_path = cache_dir / cache_key / artifact
        if svg_path.exists():
            try:
                svg_content = svg_path.read_text()
                # Basic validation that it's an SVG
                if "<svg" in svg_content and "</svg>" in svg_content:
                    html_parts.append('<div class="artifact-preview">')
                    html_parts.append(svg_content)
                    html_parts.append("</div>")
                else:
                    # Fallback to img tag if not valid SVG
                    html_parts.append('<div class="artifact-preview">')
                    html_parts.append(
                        f'<img src="artifacts/{cell_id}/{artifact}" alt="{artifact}">'
                    )
                    html_parts.append("</div>")
            except Exception:
                # Fallback to img tag on error
                html_parts.append('<div class="artifact-preview">')
                html_parts.append(
                    f'<img src="artifacts/{cell_id}/{artifact}" alt="{artifact}">'
                )
                html_parts.append("</div>")
    elif artifact.endswith(".csv"):
        # Read and convert CSV to HTML table
        csv_path = cache_dir / cache_key / artifact
        if csv_path.exists():
            try:
                import csv
                with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    rows = list(reader)

                if rows:
                    html_parts.append('<div class="artifact-preview artifact-csv">')
                    html_parts.append('<table class="csv-table">')

                    # First row as header
                    if len(rows) > 0:
                        html_parts.append('<thead><tr>')
                        for cell in rows[0]:
                            html_parts.append(f'<th>{html.escape(cell)}</th>')
                        html_parts.append('</tr></thead>')

                    # Remaining rows as data
                    if len(rows) > 1:
                        html_parts.append('<tbody>')
                        for row in rows[1:]:
                            html_parts.append('<tr>')
                            for cell in row:
                                html_parts.append(f'<td>{html.escape(cell)}</td>')
                            html_parts.append('</tr>')
                        html_parts.append('</tbody>')

                    html_parts.append('</table>')
                    html_parts.append('</div>')
            except Exception as e:
                # Show error message if CSV parsing fails
                html_parts.append('<div class="artifact-preview artifact-csv-error">')
                html_parts.append(f'<p>Error rendering CSV: {html.escape(str(e))}</p>')
                html_parts.append('</div>')

    return "\n".join(html_parts)


def render_cell(
    cell: CodeCell, result: ExecutionResult, highlighted_code: str, work_dir: Path, config: DocumentConfig
) -> str:
    """Render a single cell as HTML."""
    cell_class = "cell"
    if not result.success:
        cell_class += " cell-failed"
    if cell.commented:
        cell_class += " cell-commented"

    html_parts = [f'<div class="{cell_class}" id="cell-{cell.id}">']

    # Cell header
    header_parts = [f"Cell: {cell.id}"]
    if cell.deps:
        header_parts.append(f'deps: {", ".join(cell.deps)}')
    if result.duration:
        header_parts.append(f"{result.duration:.2f}s")
    if not result.success:
        header_parts.append("FAILED")
    if cell.commented:
        header_parts.append("COMMENTED")

    # Check if UV logs exist
    uv_logs = ""
    if result.stderr:
        uv_logs, _ = split_uv_install_logs(result.stderr)

    # Add collapse indicators to header
    code_indicator = "▶" if cell.collapse_code else "▼"
    output_indicator = "▶" if cell.collapse_output else "▼"

    html_parts.append(f'<div class="cell-header">')
    html_parts.append(f'<span class="collapse-indicators">')
    html_parts.append(
        f'<span onclick="toggleCode(\'{cell.id}\')" style="cursor: pointer;">{code_indicator} code</span> '
    )
    html_parts.append(
        f'<span onclick="toggleOutput(\'{cell.id}\')" style="cursor: pointer;">{output_indicator} output</span>'
    )

    # Always add UV logs indicator, gray out if no logs
    if uv_logs:
        html_parts.append(
            f' <span id="uv-indicator-{cell.id}" onclick="toggleUvLogsFromHeader(\'{cell.id}\')" style="cursor: pointer;">▶ uv-logs</span>'
        )
    else:
        html_parts.append(
            f' <span id="uv-indicator-{cell.id}" style="cursor: default; opacity: 0.3;">▶ uv-logs</span>'
        )

    html_parts.append(f"</span> | ")
    html_parts.append(" | ".join(header_parts))
    html_parts.append(
        f' | <button class="run-btn" onclick="runCell(\'{cell.id}\')">▶ run</button>'
    )
    html_parts.append(
        f'<button class="copy-btn" onclick="copyCell(\'{cell.id}\')">Copy</button>'
    )
    html_parts.append(
        f'<a href="cells/{cell.id}.py" target="_blank" class="raw-btn">Raw</a>'
    )
    html_parts.append("</div>")

    # Cell code - handle collapse state
    code_class = "cell-code"
    if cell.collapse_code:
        code_class += " collapsed"
    html_parts.append(f'<div id="code-{cell.id}" class="{code_class}" data-lines="{len(cell.code.splitlines())}">')
    if config.show_line_numbers:
        # Two-column layout: clickable line numbers + highlighted code
        html_parts.append('<div class="highlight-with-lines">')
        # Line numbers column
        html_parts.append(f'<div class="line-numbers" id="lines-{cell.id}">')
        for i, _ in enumerate(cell.code.splitlines(), start=1):
            html_parts.append(
                f'<a class="line-number" data-cell="{cell.id}" data-line="{i}" href="#cell-{cell.id}" '
                f'onclick="event.preventDefault(); selectCellLine(\'{cell.id}\', {i}, true);">{i}</a>'
            )
        html_parts.append('</div>')
        # Code area wrap (for overlay positioning)
        html_parts.append('<div class="code-wrap">')
        html_parts.append(highlighted_code)
        html_parts.append(f'<div class="code-line-highlight" id="line-highlight-{cell.id}"></div>')
        html_parts.append('</div>')
        html_parts.append('</div>')
    else:
        # No line numbers; still add a wrap with overlay for highlighting support
        html_parts.append('<div class="code-wrap">')
        html_parts.append(highlighted_code)
        html_parts.append(f'<div class="code-line-highlight" id="line-highlight-{cell.id}"></div>')
        html_parts.append('</div>')
    html_parts.append('</div>')

    # Cell output - handle collapse state
    output_class = "cell-output"
    if cell.collapse_output:
        output_class += " collapsed"
    html_parts.append(f'<div id="output-{cell.id}" class="{output_class}">')

    # Process stdout for inline artifact references
    cache_dir = work_dir / ".uvnote" / "cache"
    embedded_artifacts = set()

    if result.stdout:
        stdout_content = result.stdout

        # Find and replace inline artifact references: ![artifact:filename]
        if result.artifacts:
            import re
            pattern = r'!\[artifact:([^\]]+)\]'

            def replace_artifact_ref(match):
                artifact_name = match.group(1)
                if artifact_name in result.artifacts:
                    embedded_artifacts.add(artifact_name)
                    return render_artifact_preview(artifact_name, cell.id, cache_dir, result.cache_key)
                return match.group(0)  # Keep original if artifact not found

            stdout_content = re.sub(pattern, replace_artifact_ref, stdout_content)

        if getattr(result, "is_html", False):
            html_parts.append(f'<div class="cell-stdout">{stdout_content}</div>')
        else:
            html_parts.append(
                f'<div class="cell-stdout">{html.escape(stdout_content)}</div>'
            )

    if result.stderr:
        uv_logs, regular_stderr = split_uv_install_logs(result.stderr)

        # Add UV install logs in a collapsed box if present
        if uv_logs:
            html_parts.append(f'<div class="uv-install-logs" id="uv-logs-{cell.id}">')
            html_parts.append(
                f'<div class="uv-logs-header" onclick="toggleUvLogs(this)">▶ UV Install Logs</div>'
            )
            html_parts.append(
                f'<div class="uv-logs-content" style="display: none;">'
            )
            html_parts.append(html.escape(uv_logs))
            html_parts.append('</div>')
            html_parts.append('</div>')

        # Add regular stderr if present
        if regular_stderr:
            html_parts.append(
                f'<div class="cell-stderr">{html.escape(regular_stderr)}</div>'
            )

    if result.artifacts:
        html_parts.append('<div class="cell-artifacts">')
        html_parts.append("<h4>Artifacts:</h4>")

        for artifact in result.artifacts:
            html_parts.append(
                f'<a href="artifacts/{cell.id}/{artifact}" class="artifact" target="_blank">{artifact}</a>'
            )

        # Embed previews for artifacts not already embedded inline
        for artifact in result.artifacts:
            if artifact not in embedded_artifacts:
                preview_html = render_artifact_preview(artifact, cell.id, cache_dir, result.cache_key)
                if preview_html:
                    html_parts.append(preview_html)

        html_parts.append("</div>")

    html_parts.append("</div>")
    html_parts.append("</div>")

    return "\n".join(html_parts)


def process_inline_artifact_refs(content: str, results: List[ExecutionResult], work_dir: Path) -> str:
    """Process inline artifact references in markdown content."""
    import re

    cache_dir = work_dir / ".uvnote" / "cache"

    # Build mapping of artifact name to (cell_id, cache_key)
    artifact_map = {}
    for result in results:
        if result.artifacts:
            for artifact in result.artifacts:
                artifact_map[artifact] = (result.cell_id, result.cache_key)

    def replace_artifact_ref(match):
        artifact_name = match.group(1)
        if artifact_name in artifact_map:
            cell_id, cache_key = artifact_map[artifact_name]
            return render_artifact_preview(artifact_name, cell_id, cache_dir, cache_key)
        return match.group(0)  # Keep original if artifact not found

    pattern = r'!\[artifact:([^\]]+)\]'
    return re.sub(pattern, replace_artifact_ref, content)


def generate_html(
    markdown_content: str,
    config: DocumentConfig,
    cells: List[CodeCell],
    results: List[ExecutionResult],
    output_path: Path,
    work_dir: Path,
    parent_dir: Optional[str] = None,
) -> None:
    """Generate static HTML from markdown content and execution results.

    Args:
        parent_dir: Relative path to parent directory for back button (e.g., "../")
    """

    # Extract content without frontmatter for processing
    from .parser import parse_frontmatter

    _, content_without_frontmatter = parse_frontmatter(markdown_content)

    # Convert markdown to HTML (excluding code blocks)
    md = markdown.Markdown(extensions=["extra", "codehilite"])

    # Prepare cell data for template
    results_by_id = {r.cell_id: r for r in results}
    cells_by_id = {cell.id: cell for cell in cells}

    # Process markdown, replacing code blocks with rendered cells
    lines = content_without_frontmatter.splitlines()
    new_lines = []
    i = 0

    while i < len(lines):
        # Check if this line starts a Python code block
        if lines[i].strip().startswith("```python"):
            # Find matching cell
            cell_found = False
            for cell in cells:
                if cell.line_start == i + 1:  # Fence at i=4, first code line at i+1=5
                    # Render the cell HTML here
                    result = results_by_id.get(cell.id)
                    if result:
                        highlighted_code = highlight_code(cell.code, config)
                        cell_html = render_cell(
                            cell, result, highlighted_code, work_dir, config
                        )
                        new_lines.append(cell_html)
                    cell_found = True
                    break

            # Skip until we find the closing ```
            while i < len(lines) and not lines[i].strip() == "```":
                i += 1
            i += 1  # Skip the closing ```
        else:
            new_lines.append(lines[i])
            i += 1

    # Convert to HTML
    clean_content = "\n".join(new_lines)

    # Process inline artifact references in markdown content
    clean_content = process_inline_artifact_refs(clean_content, results, work_dir)

    content_html = md.convert(clean_content)

    # Setup Jinja2 environment
    # Prefer loading from file-based templates for maintainability; fallback to inline string
    templates_path = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(str(templates_path)))
    try:
        template = env.get_template("base.html.j2")
    except TemplateNotFound:
        # Fallback to inline template for backwards compatibility
        env_fallback = Environment(loader=BaseLoader())
        template = env_fallback.from_string(HTML_TEMPLATE)

    # Get Pygments CSS for both themes
    # Dark theme CSS (use configured syntax theme)
    dark_formatter = HtmlFormatter(style=config.syntax_theme)
    dark_css = dark_formatter.get_style_defs('[data-theme="dark"] .highlight')

    # Light theme CSS (use a light-friendly theme)
    light_formatter = HtmlFormatter(style="default")
    light_css = light_formatter.get_style_defs('[data-theme="light"] .highlight')

    # Combine both CSS
    pygments_css = f"{light_css}\n\n{dark_css}"

    # Determine title
    title = config.title if config.title else output_path.stem

    # Get system information
    system_info = get_system_info()

    # Render HTML
    html = template.render(
        title=title,
        config=config,
        content=content_html,
        pygments_css=pygments_css,
        system_info=system_info,
        parent_dir=parent_dir,
    )

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write HTML atomically to avoid clients reading a partially-written file
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        f.write(html)
        try:
            f.flush()
            os.fsync(f.fileno())
        except Exception:
            # fsync best-effort; ignore on platforms/filesystems that don't support it
            pass
    os.replace(tmp_path, output_path)

    # Copy artifacts to output directory
    artifacts_dir = output_path.parent / "artifacts"
    cache_dir = work_dir / ".uvnote" / "cache"

    for result in results:
        if result.artifacts:
            result_cache_dir = cache_dir / result.cache_key
            target_dir = artifacts_dir / result.cell_id
            target_dir.mkdir(parents=True, exist_ok=True)

            for artifact in result.artifacts:
                src = result_cache_dir / artifact
                dst = target_dir / artifact
                if src.exists():
                    if src.is_file():
                        # Copy atomically to avoid serving partial files
                        tmp_dst = dst.with_suffix(dst.suffix + ".tmp")
                        shutil.copy2(src, tmp_dst)
                        tmp_dst.replace(dst)
                    else:
                        shutil.copytree(src, dst, dirs_exist_ok=True)
