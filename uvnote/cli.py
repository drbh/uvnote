"""CLI commands for uvnote."""

import shutil
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Optional

import click
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from .executor import execute_cells, execute_cells_cancellable
from .generator import generate_html
from .parser import parse_markdown, validate_cells
from .server import Broadcaster, create_app
from .cache import evict_to_target, get_cache_cap_bytes, get_total_size_bytes, init_db
from .logging_config import setup_logging, get_logger
from .rebuild_queue import RebuildQueueManager


def resolve_file_path(file_input: str) -> Path:
    """
    Resolve file path, downloading from URL if it starts with https.

    Args:
        file_input: File path or HTTPS URL

    Returns:
        Path to local file (possibly in temp directory)
    """
    if file_input.startswith("https://"):
        logger = get_logger("cli")
        logger.info(f"Downloading file from {file_input}")

        try:
            # Create temp directory for downloaded files
            temp_dir = Path(tempfile.mkdtemp(prefix="uvnote_"))

            # Extract filename from URL or use default
            url_path = file_input.split("/")[-1]
            if not url_path or not url_path.endswith(".md"):
                filename = "downloaded.md"
            else:
                filename = url_path

            temp_file = temp_dir / filename

            # Download the file
            urllib.request.urlretrieve(file_input, temp_file)
            logger.info(f"Downloaded to {temp_file}")

            return temp_file

        except Exception as e:
            raise click.ClickException(f"Failed to download {file_input}: {e}")

    # Regular file path
    return Path(file_input)


class MarkdownHandler(FileSystemEventHandler):
    """File system event handler for markdown files."""

    def __init__(self, file_path: Path, callback):
        self.file_path = file_path
        self.callback = callback
        self.last_modified = 0

    def on_modified(self, event):
        if event.is_directory:
            return

        if Path(event.src_path).resolve() == self.file_path.resolve():
            # Debounce rapid file changes
            now = time.time()
            if now - self.last_modified > 0.1:
                self.last_modified = now
                self.callback()


@click.group()
@click.version_option()
@click.option(
    "--log-file",
    envvar="UVNOTE_LOG_FILE",
    default=None,
    help="Path to log file (default: console only)",
)
@click.option(
    "--log-level",
    envvar="UVNOTE_LOG_LEVEL",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    help="Logging level",
)
@click.option(
    "--no-console-log", is_flag=True, help="Disable console logging (file only)"
)
@click.pass_context
def main(ctx, log_file, log_level, no_console_log):
    """uvnote: Stateless, deterministic notebooks with uv and Markdown."""
    # Set up logging
    setup_logging(
        log_file=log_file, log_level=log_level, console_output=not no_console_log
    )
    ctx.ensure_object(dict)
    ctx.obj["logger"] = get_logger("cli")


@main.command()
@click.option(
    "--name",
    "-n",
    default="note.md",
    help="Name of the markdown file to create",
)
def init(name: str):
    """Initialize a new uvnote project with a markdown file containing a Python block."""
    file_path = Path(name)

    if file_path.exists():
        click.echo(f"Error: {name} already exists", err=True)
        return 1

    # Create the markdown file with a Python code block
    content = """# New Note

This is a new uvnote markdown file.

```python
# Your Python code here
print("Hello, uvnote!")
```
"""

    try:
        file_path.write_text(content)
        click.echo(f"Created {name}")
        return 0
    except Exception as e:
        click.echo(f"Error creating {name}: {e}", err=True)
        return 1


@main.command()
@click.argument("file")
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output directory (default: site/)",
)
@click.option("--no-cache", is_flag=True, help="Disable caching")
@click.option("--rerun", is_flag=True, help="Force rerun even if cached")
@click.option(
    "--dependencies",
    is_flag=True,
    help="Force rerun of all cells and their dependencies",
)
@click.option(
    "--incremental", is_flag=True, help="Update HTML after each cell execution"
)
def build(
    file: str,
    output: Optional[Path],
    no_cache: bool,
    rerun: bool,
    dependencies: bool,
    incremental: bool,
):
    """Build static HTML from markdown file."""

    # Resolve file path (download if URL)
    resolved_file = resolve_file_path(file)

    if output is None:
        output = Path("site")

    assert output is not None  # Help type checker understand output is not None

    work_dir = resolved_file.parent

    # Read markdown file
    try:
        with open(resolved_file) as f:
            content = f.read()
    except Exception as e:
        click.echo(f"Error reading file: {e}", err=True)
        return 1

    # Parse cells
    try:
        config, cells = parse_markdown(content)
        validate_cells(cells)
    except Exception as e:
        click.echo(f"Error parsing markdown: {e}", err=True)
        return 1

    if not cells:
        click.echo("No Python code cells found")
        return 0

    click.echo(f"Found {len(cells)} code cells")

    # Calculate dependencies if needed
    force_rerun_cells = None
    if dependencies:
        from .executor import find_all_dependencies

        # For build command, we rerun ALL cells if dependencies flag is used
        all_cell_ids = {cell.id for cell in cells}
        force_rerun_cells = all_cell_ids
        logger = get_logger("cli")
        logger.info(f"dependencies_mode: cells={len(force_rerun_cells)} rerun=all")
        logger.info("")

    # Set up incremental callback if needed
    output_file = output / f"{resolved_file.stem}.html"
    incremental_callback = None

    if incremental:
        # Prepare initial HTML with cached results where possible
        from .executor import (
            ExecutionResult,
            check_all_cells_staleness,
        )

        staleness_summary = check_all_cells_staleness(cells, work_dir)

        logger = get_logger("cli")
        logger.info("incremental_mode: preparing initial HTML with cached results")
        initial_results = []
        cached_initial_by_id = {}

        import json

        for cell in cells:
            status = staleness_summary["cell_status"][cell.id]
            # If --rerun is used, treat all cells as stale for initial display
            if not status["stale"] and not rerun:
                cache_key = status["cache_key"]
                cache_dir = work_dir / ".uvnote" / "cache" / cache_key
                result_file = cache_dir / "result.json"
                try:
                    with open(result_file) as f:
                        cached_result = json.load(f)

                    artifacts = []
                    if cache_dir.exists():
                        for item in cache_dir.iterdir():
                            if item.name not in {
                                "result.json",
                                "stdout.txt",
                                "stderr.txt",
                            }:
                                artifacts.append(str(item.relative_to(cache_dir)))

                    result = ExecutionResult(
                        cell_id=cell.id,
                        success=cached_result.get("success", True),
                        stdout=cached_result.get("stdout", ""),
                        stderr=cached_result.get("stderr", ""),
                        duration=cached_result.get("duration", 0.0),
                        artifacts=artifacts,
                        cache_key=cache_key,
                    )
                    initial_results.append(result)
                    cached_initial_by_id[cell.id] = result
                    logger.info(f"  {cell.id}=cached")
                except Exception:
                    placeholder = ExecutionResult(
                        cell_id=cell.id,
                        success=True,
                        stdout='<div class="loading-spinner"></div><div class="loading-skeleton"></div>',
                        stderr="",
                        duration=0.0,
                        artifacts=[],
                        cache_key="loading",
                        is_html=True,
                    )
                    initial_results.append(placeholder)
                    logger.info(f"  {cell.id}=loading (cache_error)")
            else:
                placeholder = ExecutionResult(
                    cell_id=cell.id,
                    success=True,
                    stdout='<div class="loading-spinner"></div><div class="loading-skeleton"></div>',
                    stderr="",
                    duration=0.0,
                    artifacts=[],
                    cache_key="loading",
                    is_html=True,
                )
                initial_results.append(placeholder)
                reason = "rerun" if not status["stale"] and rerun else status["reason"]
                logger.info(f"  {cell.id}=loading ({reason})")

        def update_html(partial_results):
            try:
                mixed_results = []
                completed_cell_ids = {r.cell_id for r in partial_results}

                for cell in cells:
                    if cell.id in completed_cell_ids:
                        result = next(
                            r for r in partial_results if r.cell_id == cell.id
                        )
                        mixed_results.append(result)
                    elif cell.id in cached_initial_by_id:
                        mixed_results.append(cached_initial_by_id[cell.id])
                    else:
                        placeholder = ExecutionResult(
                            cell_id=cell.id,
                            success=True,
                            stdout='<div class="loading-spinner"></div><div class="loading-skeleton"></div>',
                            stderr="",
                            duration=0.0,
                            artifacts=[],
                            cache_key="loading",
                            is_html=True,
                        )
                        mixed_results.append(placeholder)

                generate_html(
                    content, config, cells, mixed_results, output_file, work_dir
                )
                logger.info(f"  incremental_update: {output_file}")
            except Exception as e:
                logger.error(f"  incremental_error: {e}")

        incremental_callback = update_html

        try:
            generate_html(
                content, config, cells, initial_results, output_file, work_dir
            )
            logger.info(f"  initial: {output_file}")
        except Exception as e:
            click.echo(f"Error generating initial HTML: {e}", err=True)

    # Execute cells
    try:
        results = execute_cells(
            cells,
            work_dir,
            use_cache=not (no_cache or rerun),
            force_rerun_cells=force_rerun_cells,
            incremental_callback=incremental_callback,
        )
    except Exception as e:
        click.echo(f"Error executing cells: {e}", err=True)
        return 1

    # Check for failures
    failed_cells = [r for r in results if not r.success]
    if failed_cells:
        click.echo(f"Warning: {len(failed_cells)} cells failed execution")
        for result in failed_cells:
            click.echo(
                f"  - {result.cell_id}: {result.stderr.split()[0] if result.stderr else 'Unknown error'}"
            )

    # Generate final HTML (only if not incremental, since incremental already generated it)
    if not incremental:
        try:
            generate_html(content, config, cells, results, output_file, work_dir)
            click.echo(f"Generated: {output_file}")
        except Exception as e:
            click.echo(f"Error generating HTML: {e}", err=True)
            return 1
    else:
        click.echo(f"Final: {output_file}")

    # Copy cell files to output directory for URL access
    try:
        cells_src = work_dir / ".uvnote" / "cells"
        cells_dst = output / "cells"

        if cells_src.exists():
            # Remove existing cells directory if it exists
            if cells_dst.exists():
                shutil.rmtree(cells_dst)

            # Copy the entire cells directory
            shutil.copytree(cells_src, cells_dst)
            click.echo(f"Copied cell files to: {cells_dst}")
        else:
            click.echo("No cell files found to copy")
    except Exception as e:
        click.echo(f"Warning: Failed to copy cell files: {e}", err=True)

    return 0


@main.command()
@click.argument("file", type=str)
@click.option("--cell", help="Cell ID to run (if not specified, runs all cells)")
@click.option("--no-cache", is_flag=True, help="Disable caching")
@click.option("--rerun", is_flag=True, help="Force rerun even if cached")
@click.option(
    "--dependencies", is_flag=True, help="Force rerun of cell and all its dependencies"
)
@click.option(
    "--check", is_flag=True, help="Check which cells are stale without executing"
)
def run(
    file: str,
    cell: Optional[str],
    no_cache: bool,
    rerun: bool,
    dependencies: bool,
    check: bool,
):
    """Run cells from markdown file."""

    # Resolve file path (download if URL)
    resolved_file = resolve_file_path(file)
    work_dir = resolved_file.parent

    # Read and parse markdown
    try:
        with open(resolved_file) as f:
            content = f.read()

        config, cells = parse_markdown(content)
        validate_cells(cells)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        return 1

    if not cells:
        click.echo("No Python code cells found")
        return 0

    click.echo(f"Found {len(cells)} code cells")

    # Handle --check flag
    if check:
        from .executor import check_all_cells_staleness, check_cell_staleness

        if cell is not None:
            # Check single cell
            target_cell = None
            for c in cells:
                if c.id == cell:
                    target_cell = c
                    break

            if target_cell is None:
                click.echo(f"Cell '{cell}' not found")
                available = [c.id for c in cells]
                if available:
                    click.echo(f"Available cells: {', '.join(available)}")
                return 1

            # For single cell check, we need to consider environment variables too
            # This is a simplified approach - we can't know the exact env vars without running dependencies
            # But we can check if the cell has dependencies and warn about this limitation
            if target_cell.needs:
                # We need to simulate the environment variables that would be present
                from .executor import check_all_cells_staleness

                # Get the full staleness check to get proper env vars
                summary = check_all_cells_staleness(cells, work_dir)
                status = summary["cell_status"][cell]
            else:
                status = check_cell_staleness(target_cell, work_dir)

            logger = get_logger("cli")
            logger.info("staleness_check:")
            logger.info(f"  cell={cell}")
            logger.info(f"  stale={str(status['stale']).lower()}")
            logger.info(f"  reason={status['reason']}")
            if not status["stale"]:
                logger.info(f"  cached_duration={status['duration']:.2f}s")
                logger.info(f"  cached_success={str(status['success']).lower()}")
        else:
            # Check all cells
            summary = check_all_cells_staleness(cells, work_dir)

            logger = get_logger("cli")
            logger.info("staleness_check:")
            logger.info(f"  total_cells={summary['total_cells']}")
            logger.info(f"  stale_count={summary['stale_count']}")
            logger.info(f"  cached_count={summary['cached_count']}")
            if summary["cyclic_count"] > 0:
                logger.info(f"  cyclic_count={summary['cyclic_count']}")

            if summary["stale_cells"]:
                logger.info(f"  stale_cells={','.join(summary['stale_cells'])}")
            if summary["cached_cells"]:
                logger.info(f"  cached_cells={','.join(summary['cached_cells'])}")
            if summary["cyclic_cells"]:
                logger.info(f"  cyclic_cells={','.join(summary['cyclic_cells'])}")

            logger.info("\ndetailed_status:")
            for cell_id in summary["execution_order"]:
                status = summary["cell_status"][cell_id]
                stale_str = str(status["stale"]).lower()
                logger.info(f"  {cell_id}=stale={stale_str} reason={status['reason']}")

            for cell_id in summary["cyclic_cells"]:
                status = summary["cell_status"][cell_id]
                stale_str = str(status["stale"]).lower()
                logger.info(f"  {cell_id}=stale={stale_str} reason={status['reason']}")

        return 0

    if cell is not None:
        # Single cell mode
        target_cell = None
        for c in cells:
            if c.id == cell:
                target_cell = c
                break

        if target_cell is None:
            click.echo(f"Cell '{cell}' not found")
            available = [c.id for c in cells]
            if available:
                click.echo(f"Available cells: {', '.join(available)}")
            return 1

        # Calculate dependencies if needed for single cell
        if dependencies:
            from .executor import find_all_dependencies, execute_cells

            force_rerun_cells = find_all_dependencies(cells, target_cell.id)
            logger = get_logger("cli")
            logger.info(
                f"dependencies_mode: cells={len(force_rerun_cells)} rerun={','.join(sorted(force_rerun_cells))}"
            )
            logger.info("")

            # Filter cells to only those needed and execute them all in dependency order
            dependency_cells = [c for c in cells if c.id in force_rerun_cells]

            try:
                results = execute_cells(
                    dependency_cells,
                    work_dir,
                    use_cache=not (no_cache or rerun),
                    force_rerun_cells=force_rerun_cells,
                )
                # Find the result for our target cell
                result = next(r for r in results if r.cell_id == target_cell.id)
            except Exception as e:
                click.echo(f"Error executing dependencies: {e}", err=True)
                return 1
        else:
            # Execute just the single cell
            from .executor import execute_cell

            try:
                result = execute_cell(
                    target_cell, work_dir, use_cache=not (no_cache or rerun)
                )
            except Exception as e:
                click.echo(f"Error executing cell: {e}", err=True)
                return 1

        # Show single cell results
        if result.stdout:
            click.echo("STDOUT:")
            click.echo(result.stdout)

        if result.stderr:
            click.echo("STDERR:")
            click.echo(result.stderr)

        if result.artifacts:
            click.echo(f"Artifacts: {', '.join(result.artifacts)}")

        click.echo(f"Duration: {result.duration:.2f}s")

        return 0 if result.success else 1

    else:
        # All cells mode (like build but without HTML generation)
        force_rerun_cells = None
        if dependencies:
            # For all cells mode, rerun all cells if dependencies flag is used
            all_cell_ids = {cell.id for cell in cells}
            force_rerun_cells = all_cell_ids
            logger = get_logger("cli")
            logger.info(f"dependencies_mode: cells={len(force_rerun_cells)} rerun=all")
            logger.info("")

        # Execute all cells
        try:
            from .executor import execute_cells

            results = execute_cells(
                cells,
                work_dir,
                use_cache=not (no_cache or rerun),
                force_rerun_cells=force_rerun_cells,
            )
        except Exception as e:
            click.echo(f"Error executing cells: {e}", err=True)
            return 1

        # Check for failures
        failed_cells = [r for r in results if not r.success]
        if failed_cells:
            click.echo(f"Warning: {len(failed_cells)} cells failed execution")
            for result in failed_cells:
                click.echo(
                    f"  - {result.cell_id}: {result.stderr.split()[0] if result.stderr else 'Unknown error'}"
                )

        return 0 if not failed_cells else 1


@main.command("build-loading")
@click.argument("file", type=str)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output directory (default: site/)",
)
def build_loading(file: str, output: Optional[Path]):
    """Build HTML with loading placeholders for stale cells."""

    # Resolve file path (download if URL)
    resolved_file = resolve_file_path(file)

    if output is None:
        output = Path("site")

    assert output is not None  # Help type checker understand output is not None

    work_dir = resolved_file.parent

    # Read markdown file
    try:
        with open(resolved_file) as f:
            content = f.read()
    except Exception as e:
        click.echo(f"Error reading file: {e}", err=True)
        return 1

    # Parse cells
    try:
        from .parser import parse_markdown, validate_cells

        config, cells = parse_markdown(content)
        validate_cells(cells)
    except Exception as e:
        click.echo(f"Error parsing markdown: {e}", err=True)
        return 1

    if not cells:
        click.echo("No Python code cells found")
        return 0

    click.echo(f"Found {len(cells)} code cells")

    # Check staleness of all cells
    from .executor import check_all_cells_staleness, ExecutionResult

    staleness_summary = check_all_cells_staleness(cells, work_dir)

    logger = get_logger("cli")
    logger.info("staleness_check:")
    logger.info(f"  total_cells={staleness_summary['total_cells']}")
    logger.info(f"  stale_count={staleness_summary['stale_count']}")
    logger.info(f"  cached_count={staleness_summary['cached_count']}")

    # Create results: real results for cached cells, loading placeholders for stale cells
    results = []
    import json

    for cell in cells:
        cell_status = staleness_summary["cell_status"][cell.id]

        if not cell_status["stale"]:
            # Load actual cached result
            cache_key = cell_status["cache_key"]
            cache_dir = work_dir / ".uvnote" / "cache" / cache_key
            result_file = cache_dir / "result.json"

            try:
                with open(result_file) as f:
                    cached_result = json.load(f)

                # Find artifacts
                artifacts = []
                if cache_dir.exists():
                    for item in cache_dir.iterdir():
                        if item.name not in {"result.json", "stdout.txt", "stderr.txt"}:
                            artifacts.append(str(item.relative_to(cache_dir)))

                result = ExecutionResult(
                    cell_id=cell.id,
                    success=cached_result["success"],
                    stdout=cached_result["stdout"],
                    stderr=cached_result["stderr"],
                    duration=cached_result["duration"],
                    artifacts=artifacts,
                    cache_key=cache_key,
                )
                logger.info(f"  {cell.id}=cached")
            except Exception:
                # If we can't load cached result, treat as loading
                result = ExecutionResult(
                    cell_id=cell.id,
                    success=True,
                    stdout='<div class="loading-spinner"></div><div class="loading-skeleton"></div>',
                    stderr="",
                    duration=0.0,
                    artifacts=[],
                    cache_key="loading",
                    is_html=True,
                )
                logger.info(f"  {cell.id}=loading (cache_error)")
        else:
            # Create loading placeholder
            result = ExecutionResult(
                cell_id=cell.id,
                success=True,
                stdout='<div class="loading-spinner"></div><div class="loading-skeleton"></div>',
                stderr="",
                duration=0.0,
                artifacts=[],
                cache_key="loading",
                is_html=True,
            )
            logger.info(f"  {cell.id}=loading ({cell_status['reason']})")

        results.append(result)

    # Generate HTML
    output_file = output / f"{resolved_file.stem}.html"
    try:
        from .generator import generate_html

        generate_html(content, config, cells, results, output_file, work_dir)
        click.echo(f"Generated: {output_file}")
    except Exception as e:
        click.echo(f"Error generating HTML: {e}", err=True)
        return 1

    # Copy cell files to output directory for URL access
    try:
        cells_src = work_dir / ".uvnote" / "cells"
        cells_dst = output / "cells"

        if cells_src.exists():
            # Remove existing cells directory if it exists
            if cells_dst.exists():
                shutil.rmtree(cells_dst)

            # Copy the entire cells directory
            shutil.copytree(cells_src, cells_dst)
            click.echo(f"Copied cell files to: {cells_dst}")
        else:
            click.echo("No cell files found to copy")
    except Exception as e:
        click.echo(f"Warning: Failed to copy cell files: {e}", err=True)

    return 0


@main.command()
@click.argument("file", type=str)
def graph(file: str):
    """Show dependency graph for markdown file."""

    # Resolve file path (download if URL)
    resolved_file = resolve_file_path(file)

    # Read and parse markdown
    try:
        with open(resolved_file) as f:
            content = f.read()

        from .parser import parse_markdown, validate_cells

        config, cells = parse_markdown(content)
        validate_cells(cells)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        return 1

    if not cells:
        click.echo("No Python code cells found")
        return 0

    click.echo(f"Found {len(cells)} code cells")

    # Print dependency graph
    from .executor import print_dependency_matrix

    print_dependency_matrix(cells)

    # Print execution order
    from .executor import execute_cells

    # We need to get the execution order without actually executing
    # Let's extract the topo_sort logic
    def get_execution_order(cells):
        from collections import deque
        from typing import Dict, List, Set, Tuple

        def topo_sort(cells) -> Tuple[List[str], Set[str]]:
            # Graph: edge from need -> cell.id
            ids = {c.id for c in cells}
            indeg: Dict[str, int] = {cid: 0 for cid in ids}
            adj: Dict[str, List[str]] = {cid: [] for cid in ids}

            for c in cells:
                for need in c.needs:
                    if need in ids:
                        adj[need].append(c.id)
                        indeg[c.id] += 1

            # Kahn's algorithm
            order: List[str] = []
            q = deque([cid for cid, d in indeg.items() if d == 0])
            while q:
                u = q.popleft()
                order.append(u)
                for v in adj.get(u, []):
                    indeg[v] -= 1
                    if indeg[v] == 0:
                        q.append(v)

            # Any nodes not in order are part of cycles
            leftover = set(ids) - set(order)
            return order, leftover

        return topo_sort(cells)

    order, cyclic = get_execution_order(cells)

    logger = get_logger("cli")
    logger.info("execution_graph:")
    logger.info(f"  cells={len(order)}")
    logger.info(f"  order={' -> '.join(order)}")
    if cyclic:
        logger.warning(f"  warning=cyclic_dependencies cells={','.join(cyclic)}")

    return 0


@main.command()
@click.option("--all", is_flag=True, help="Clean all cache and site files")
def clean(all: bool):
    """Clear cache and site files."""

    work_dir = Path.cwd()

    if all or click.confirm("Remove .uvnote directory?"):
        uvnote_dir = work_dir / ".uvnote"
        if uvnote_dir.exists():
            shutil.rmtree(uvnote_dir)
            click.echo("Removed .uvnote/")

    if all or click.confirm("Remove site directory?"):
        site_dir = work_dir / "site"
        if site_dir.exists():
            shutil.rmtree(site_dir)
            click.echo("Removed site/")


@main.command("cache-prune")
@click.option("--size", help="Target size cap like 5GB, 500MB. Defaults to env/10GB.")
def cache_prune(size: str | None):
    """Prune cache to target size using LRU eviction."""
    work_dir = Path.cwd()
    init_db(work_dir)

    def parse_size(s: str | None, default_bytes: int) -> int:
        if not s:
            return default_bytes
        val = s.strip().lower()
        try:
            if val.endswith(("kb", "k")):
                return int(float(val.rstrip("kbk"))) * 1024
            if val.endswith(("mb", "m")):
                return int(float(val.rstrip("mbm"))) * 1024 * 1024
            if val.endswith(("gb", "g")):
                return int(float(val.rstrip("gbg"))) * 1024 * 1024 * 1024
            return int(val)
        except ValueError:
            return default_bytes

    current = get_total_size_bytes(work_dir)
    cap = parse_size(size, get_cache_cap_bytes())
    freed, removed = evict_to_target(work_dir, cap)
    after = get_total_size_bytes(work_dir)
    logger = get_logger("cli")
    logger.info("cache_prune:")
    logger.info(f"  before_bytes={current}")
    logger.info(f"  target_bytes={cap}")
    logger.info(f"  freed_bytes={freed}")
    logger.info(f"  removed_entries={len(removed)}")
    logger.info(f"  after_bytes={after}")


@main.command()
@click.argument("file")
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output directory (default: site/)",
)
@click.option("--host", default="localhost", help="Host to serve on")
@click.option("--port", default=8000, type=int, help="Port to serve on")
@click.option("--no-cache", is_flag=True, help="Disable caching")
def serve(file: str, output: Optional[Path], host: str, port: int, no_cache: bool):
    """Watch markdown file, rebuild on changes, and serve HTML (Flask)."""

    # Resolve file path (download if URL)
    resolved_file = resolve_file_path(file)

    if output is None:
        output = Path("site")

    assert output is not None  # Help type checker understand output is not None
    output_file = output / f"{resolved_file.stem}.html"

    broadcaster = Broadcaster()

    def rebuild(cancel_event=None):
        logger = get_logger("cli")
        logger.debug("!!!!!!!!!!!!!! Rebuilding...")
        click.echo(f"Rebuilding {resolved_file}...")
        try:
            with open(resolved_file) as f:
                content = f.read()
            config, cells = parse_markdown(content)
            validate_cells(cells)

            # Prepare initial results from cache or placeholders so nothing disappears
            from .executor import ExecutionResult, check_all_cells_staleness
            import json

            work_dir = resolved_file.parent
            staleness = check_all_cells_staleness(cells, work_dir)
            initial_results = []
            cached_initial_by_id = {}

            for cell in cells:
                status = staleness["cell_status"][cell.id]
                if status["stale"] or no_cache:
                    placeholder = ExecutionResult(
                        cell_id=cell.id,
                        success=True,
                        stdout='<div class="loading-spinner"></div><div class="loading-skeleton"></div>',
                        stderr="",
                        duration=0.0,
                        artifacts=[],
                        cache_key="loading",
                        is_html=True,
                    )
                    initial_results.append(placeholder)
                    reason = "no_cache" if no_cache else status["reason"]
                    logger.info(f"  {cell.id}=loading ({reason})")
                else:
                    cache_key = status["cache_key"]
                    cache_dir = work_dir / ".uvnote" / "cache" / cache_key
                    try:
                        with open(cache_dir / "result.json") as rf:
                            cached = json.load(rf)
                        artifacts = []
                        if cache_dir.exists():
                            for item in cache_dir.iterdir():
                                if item.name not in {
                                    "result.json",
                                    "stdout.txt",
                                    "stderr.txt",
                                }:
                                    artifacts.append(str(item.relative_to(cache_dir)))
                        res = ExecutionResult(
                            cell_id=cell.id,
                            success=cached.get("success", True),
                            stdout=cached.get("stdout", ""),
                            stderr=cached.get("stderr", ""),
                            duration=cached.get("duration", 0.0),
                            artifacts=artifacts,
                            cache_key=cache_key,
                        )
                        initial_results.append(res)
                        cached_initial_by_id[cell.id] = res
                        logger.info(f"  {cell.id}=cached")
                    except Exception:
                        placeholder = ExecutionResult(
                            cell_id=cell.id,
                            success=True,
                            stdout='<div class="loading-spinner"></div><div class="loading-skeleton"></div>',
                            stderr="",
                            duration=0.0,
                            artifacts=[],
                            cache_key="loading",
                            is_html=True,
                        )
                        initial_results.append(placeholder)
                        logger.info(f"  {cell.id}=loading (cache_error)")

            # Render initial HTML so unchanged cells remain visible
            generate_html(
                content,
                config,
                cells,
                initial_results,
                output_file,  # Use output_file which is already Path
                work_dir,
            )
            logger.info(f"  initial: {output_file}")

            # Incremental updates merge new results with existing cached/placeholder ones
            def incremental_reload_callback(partial_results):
                try:
                    mixed = []
                    completed = {r.cell_id for r in partial_results}
                    for cell in cells:
                        if cell.id in completed:
                            mixed.append(
                                next(r for r in partial_results if r.cell_id == cell.id)
                            )
                        elif cell.id in cached_initial_by_id:
                            mixed.append(cached_initial_by_id[cell.id])
                        else:
                            # Fallback placeholder
                            mixed.append(
                                ExecutionResult(
                                    cell_id=cell.id,
                                    success=True,
                                    stdout='<div class="loading-spinner"></div><div class="loading-skeleton"></div>',
                                    stderr="",
                                    duration=0.0,
                                    artifacts=[],
                                    cache_key="loading",
                                    is_html=True,
                                )
                            )
                    generate_html(
                        content,
                        config,
                        cells,
                        mixed,
                        output_file,  # Use output_file which is already Path
                        work_dir,
                    )
                    broadcaster.broadcast("incremental")
                except Exception as e:
                    logger.error(f"Incremental update error: {e}")

            # Execute cells with cancellation support
            if cancel_event:
                results = execute_cells_cancellable(
                    cells,
                    work_dir=work_dir,
                    use_cache=not no_cache,
                    incremental_callback=incremental_reload_callback,
                    cancel_event=cancel_event,
                )
            else:
                results = execute_cells(
                    cells,
                    work_dir=work_dir,
                    use_cache=not no_cache,
                    incremental_callback=incremental_reload_callback,
                )

            # Final HTML and broadcast
            generate_html(content, config, cells, results, output_file, work_dir)

            # Copy cell files to output directory for URL access
            try:
                cells_src = work_dir / ".uvnote" / "cells"
                cells_dst = output_file.parent / "cells"

                if cells_src.exists():
                    # Remove existing cells directory if it exists
                    if cells_dst.exists():
                        shutil.rmtree(cells_dst)

                    # Copy the entire cells directory
                    shutil.copytree(cells_src, cells_dst)
                    logger.info(f"  cells: {cells_dst}")
            except Exception as e:
                logger.error(f"  cells_copy_error: {e}")

            click.echo(f"Rebuilt: {output_file}")
            broadcaster.broadcast("reload")
        except Exception as e:
            click.echo(f"Rebuild failed: {e}", err=True)

    def emit_loading_state():
        """Generate and emit immediate loading HTML when file changes."""
        try:
            logger = get_logger("cli")
            logger.info("Generating immediate loading state")

            # Read and parse the current file
            with open(resolved_file) as f:
                content = f.read()
            config, cells = parse_markdown(content)
            validate_cells(cells)

            # Create loading placeholders for all cells
            from .executor import ExecutionResult

            loading_results = []
            for cell in cells:
                loading_result = ExecutionResult(
                    cell_id=cell.id,
                    success=True,
                    stdout='<div class="loading-spinner"></div><div class="loading-skeleton"></div>',
                    stderr="",
                    duration=0.0,
                    artifacts=[],
                    cache_key="loading",
                    is_html=True,
                )
                loading_results.append(loading_result)
                logger.info(f"  {cell.id}=loading (file_changed)")

            # Generate and emit loading HTML
            work_dir = resolved_file.parent
            generate_html(
                content, config, cells, loading_results, output_file, work_dir
            )
            broadcaster.broadcast("incremental")
            logger.info(f"Loading state emitted: {output_file}")

        except Exception as e:
            logger.error(f"Failed to emit loading state: {e}", exc_info=True)

    # Create rebuild queue manager
    rebuild_queue = RebuildQueueManager(
        rebuild_func=rebuild,
        loading_func=emit_loading_state,
        debounce_seconds=0.5,
        min_interval_seconds=1.0,
    )

    # Generate initial HTML with all loading states for immediate response
    logger = get_logger("cli")
    logger.info("Generating initial loading page...")
    try:
        with open(resolved_file) as f:
            content = f.read()
        config, cells = parse_markdown(content)
        validate_cells(cells)

        # Create loading placeholders for all cells
        from .executor import ExecutionResult

        loading_results = []
        for cell in cells:
            loading_result = ExecutionResult(
                cell_id=cell.id,
                success=True,
                stdout='<div class="loading-spinner"></div><div class="loading-skeleton"></div>',
                stderr="",
                duration=0.0,
                artifacts=[],
                cache_key="loading",
                is_html=True,
            )
            loading_results.append(loading_result)

        # Generate initial HTML with loading states
        work_dir = resolved_file.parent
        generate_html(content, config, cells, loading_results, output_file, work_dir)
        logger.info(f"Initial loading page created: {output_file}")
    except Exception as e:
        logger.error(f"Failed to create initial page: {e}")
        # Fall back to empty page if needed
        output.mkdir(parents=True, exist_ok=True)
        output_file.write_text("<html><body>Loading...</body></html>")

    # Watch source file with queue manager
    def on_file_change():
        rebuild_queue.request_rebuild(resolved_file)

    event_handler = MarkdownHandler(resolved_file, on_file_change)
    observer = Observer()
    observer.schedule(event_handler, str(resolved_file.parent), recursive=False)
    observer.start()

    # Start Flask app
    import threading
    import webbrowser
    from flask import jsonify  # type: ignore[import-not-found]

    app = create_app(output, output_file.name, broadcaster)

    # Schedule initial build to run after server starts
    def initial_build():
        time.sleep(0.5)  # Small delay to ensure server is up
        logger.info("Starting initial build...")
        rebuild()

    threading.Thread(target=initial_build, daemon=True).start()
    click.echo(f"Static root: {output.resolve()}")

    @app.post("/run/<cell_id>")
    def run_cell(cell_id: str):  # type: ignore[unused-variable]
        try:
            with open(file) as f:
                content = f.read()
            from uvnote.parser import parse_markdown, validate_cells
            from uvnote.executor import execute_cell

            config, cells = parse_markdown(content)
            validate_cells(cells)
            target = next((c for c in cells if c.id == cell_id), None)
            if not target:
                return jsonify({"error": f"Cell {cell_id} not found"}), 404

            # Respond immediately; execute in background without cache
            def _bg_exec():
                try:
                    execute_cell(target, Path.cwd(), use_cache=False)
                except Exception:
                    pass

            threading.Thread(target=_bg_exec, daemon=True).start()
            return jsonify({"success": True, "status": "executing", "cell_id": cell_id})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    url = f"http://{host}:{port}/{output_file.name}"
    click.echo(f"Serving at {url}")
    click.echo("Press Ctrl+C to stop")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    try:
        app.run(host=host, port=port, threaded=True, use_reloader=False, debug=False)
    except KeyboardInterrupt:
        pass
    finally:
        rebuild_queue.stop()
        observer.stop()
        observer.join()


@main.command()
@click.argument("file", type=str)
@click.option("--cell", help="Cell to export (exports all cells if not specified)")
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output directory (default: export/)",
)
def export(file: str, cell: Optional[str], output: Optional[Path]):
    """Export cell files and their dependencies to a directory."""

    if output is None:
        output = Path("export")

    assert output is not None  # Help type checker understand output is not None

    # Resolve file path (download if URL)
    resolved_file = resolve_file_path(file)

    # Read and parse markdown
    try:
        with open(resolved_file) as f:
            content = f.read()

        from .parser import parse_markdown, validate_cells

        config, cells = parse_markdown(content)
        validate_cells(cells)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        return 1

    # Create work directory
    work_dir = Path(".uvnote")
    cells_dir = work_dir / "cells"

    if not cells_dir.exists():
        click.echo(f"Error: Cell directory {cells_dir} does not exist", err=True)
        return 1

    # Determine which cells to export
    if cell:
        # Find the specific cell
        target_cell = next((c for c in cells if c.id == cell), None)
        if not target_cell:
            click.echo(f"Error: Cell '{cell}' not found", err=True)
            return 1

        # Get all dependencies for the target cell
        from .executor import find_all_dependencies

        all_deps = find_all_dependencies(cells, cell)
        cells_to_export = all_deps | {cell}

        logger = get_logger("cli")
        logger.info(f"export_target: {cell}")
        if all_deps:
            logger.info(f"dependencies: {','.join(sorted(all_deps))}")
        logger.info(f"total_files: {len(cells_to_export)}")
    else:
        # Export all cells
        cells_to_export = {c.id for c in cells}
        logger = get_logger("cli")
        logger.info(f"export_mode: all_cells")
        logger.info(f"total_files: {len(cells_to_export)}")

    # Create output directory
    output.mkdir(exist_ok=True)

    # Copy cell files
    copied_files = []
    for cell_id in cells_to_export:
        source_file = cells_dir / f"{cell_id}.py"
        target_file = output / f"{cell_id}.py"

        if source_file.exists():
            import shutil

            shutil.copy2(source_file, target_file)
            copied_files.append(cell_id)
            logger.info(f"  copied: {cell_id}.py")
        else:
            logger.warning(f"  missing: {cell_id}.py")

    logger.info(f"export_complete: {len(copied_files)} files copied to {output}")
    return 0


if __name__ == "__main__":
    main()
