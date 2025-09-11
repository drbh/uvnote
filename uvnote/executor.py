"""Cell execution engine with uv run and caching."""

import hashlib
import json
import os
import subprocess
import tempfile
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Callable

from .parser import CodeCell
from .cache import (
    init_db,
    record_access,
    record_write,
    get_cache_cap_bytes,
    evict_to_target,
    integrity_check,
    get_artifacts,
)


@dataclass
class ExecutionResult:
    """Result of executing a code cell."""

    cell_id: str
    success: bool
    stdout: str
    stderr: str
    duration: float
    artifacts: List[str]
    cache_key: str
    is_html: bool = False  # Flag to indicate if stdout contains raw HTML


def print_dependency_matrix(cells: List[CodeCell]):
    """Print dependencies in key=value format."""
    if not cells:
        return

    print("dependency_graph:")

    # Group cells by their dependencies for cleaner output
    has_deps = []
    no_deps = []

    for cell in cells:
        if cell.needs:
            has_deps.append(cell)
        else:
            no_deps.append(cell)

    # Print cells without dependencies first
    if no_deps:
        root_cells = [cell.id for cell in no_deps]
        print(f"  roots={','.join(root_cells)}")

    # Print dependency relationships
    for cell in has_deps:
        deps = ",".join(cell.needs)
        print(f"  {cell.id}={deps}")

    print()


def check_cell_staleness(
    cell: CodeCell, work_dir: Path, env_vars: Optional[Dict[str, str]] = None
) -> dict:
    """Check if a cell needs to be rerun (is stale)."""
    cache_key = generate_cache_key(cell, work_dir, env_vars)
    cache_dir = work_dir / ".uvnote" / "cache" / cache_key

    # If no cache exists, it's stale
    if not cache_dir.exists():
        return {"stale": True, "reason": "no_cache", "cache_key": cache_key}

    result_file = cache_dir / "result.json"
    if not result_file.exists():
        return {"stale": True, "reason": "missing_result", "cache_key": cache_key}

    # Cache exists and is valid
    try:
        with open(result_file) as f:
            cached_result = json.load(f)
        # Basic integrity check; if it fails, mark stale
        if not integrity_check(work_dir, cache_key):
            return {"stale": True, "reason": "integrity_failed", "cache_key": cache_key}
        record_access(work_dir, cache_key)
        return {
            "stale": False,
            "reason": "cached",
            "cache_key": cache_key,
            "duration": cached_result.get("duration", 0),
            "success": cached_result.get("success", False),
        }
    except Exception:
        return {"stale": True, "reason": "corrupt_cache", "cache_key": cache_key}


def check_all_cells_staleness(cells: List[CodeCell], work_dir: Path) -> dict:
    """Check staleness of all cells and return summary."""
    from collections import deque

    def sanitize_env_key(s: str) -> str:
        out = []
        for ch in s:
            if ch.isalnum():
                out.append(ch.upper())
            else:
                out.append("_")
        return "".join(out)

    # Get execution order
    def topo_sort(cells):
        ids = {c.id for c in cells}
        indeg = {cid: 0 for cid in ids}
        adj = {cid: [] for cid in ids}

        for c in cells:
            for need in c.needs:
                if need in ids:
                    adj[need].append(c.id)
                    indeg[c.id] += 1

        order = []
        q = deque([cid for cid, d in indeg.items() if d == 0])
        while q:
            u = q.popleft()
            order.append(u)
            for v in adj.get(u, []):
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)

        leftover = set(ids) - set(order)
        return order, leftover

    order, cyclic = topo_sort(cells)
    cells_by_id = {cell.id: cell for cell in cells}

    cell_status = {}
    stale_cells = []
    cached_cells = []
    executed = {}  # Track cache keys of executed cells

    # Check each cell in dependency order, considering environment variables
    for cell_id in order:
        cell = cells_by_id[cell_id]

        # Build per-cell env vars including inputs from dependencies
        per_cell_env = {}
        if cell.needs:
            inputs_list = []
            for need in cell.needs:
                if need in executed and not executed[need]["stale"]:
                    # Use the cache key as a proxy for the cache directory
                    dep_key = executed[need]["cache_key"]
                    dep_dir = work_dir / ".uvnote" / "cache" / dep_key
                    env_key = f"UVNOTE_INPUT_{sanitize_env_key(need)}"
                    per_cell_env[env_key] = str(dep_dir)
                    inputs_list.append(env_key)
            if inputs_list:
                per_cell_env["UVNOTE_INPUTS"] = ",".join(inputs_list)

        status = check_cell_staleness(cell, work_dir, per_cell_env)
        cell_status[cell_id] = status
        executed[cell_id] = status

        if status["stale"]:
            stale_cells.append(cell_id)
        else:
            cached_cells.append(cell_id)

    # Mark cyclic cells as stale
    for cell_id in cyclic:
        cell_status[cell_id] = {
            "stale": True,
            "reason": "cyclic_dependency",
            "cache_key": "N/A",
        }
        stale_cells.append(cell_id)

    return {
        "total_cells": len(cells),
        "stale_count": len(stale_cells),
        "cached_count": len(cached_cells),
        "cyclic_count": len(cyclic),
        "stale_cells": stale_cells,
        "cached_cells": cached_cells,
        "cyclic_cells": list(cyclic),
        "cell_status": cell_status,
        "execution_order": order,
    }


def find_all_dependencies(cells: List[CodeCell], target_cell_id: str) -> Set[str]:
    """Find all dependencies (transitive) for a given cell."""
    cells_by_id = {cell.id: cell for cell in cells}
    dependencies = set()

    def collect_deps(cell_id: str):
        if cell_id not in cells_by_id:
            return
        cell = cells_by_id[cell_id]
        for need in cell.needs:
            if need not in dependencies:
                dependencies.add(need)
                collect_deps(need)

    collect_deps(target_cell_id)
    # Include the target cell itself
    dependencies.add(target_cell_id)
    return dependencies


def generate_cache_key(
    cell: CodeCell, work_dir: Path, env_vars: Optional[Dict[str, str]] = None
) -> str:
    """Generate cache key for a cell based on code, dependencies, environment, and env lock.

    Includes:
    - cell code
    - declared deps
    - relevant env vars (dependency inputs)
    - uv.lock hash (if present)
    - python version (major.minor.micro)
    """
    import sys

    uv_lock_path = work_dir / "uv.lock"
    uv_lock_hash = None
    if uv_lock_path.exists():
        try:
            h = hashlib.sha256()
            with open(uv_lock_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            uv_lock_hash = h.hexdigest()
        except Exception:
            uv_lock_hash = None
    py_ver = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    content = {
        "code": cell.code,
        "deps": sorted(cell.deps),
        "env": sorted((env_vars or {}).items()),
        "uv_lock": uv_lock_hash or "no-lock",
        "python": py_ver,
    }
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def create_cell_script(cell: CodeCell, work_dir: Path) -> Path:
    """Create a standalone Python script for the cell with uv dependencies."""
    cells_dir = work_dir / ".uvnote" / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)

    script_path = cells_dir / f"{cell.id}.py"

    # Build the script content with uv dependencies
    script_lines = []

    if cell.deps:
        script_lines.append("# /// script")
        script_lines.append("# dependencies = [")
        for dep in cell.deps:
            script_lines.append(f'#     "{dep}",')
        script_lines.append("# ]")
        script_lines.append("# ///")
        script_lines.append("")

    script_lines.append(cell.code)

    with open(script_path, "w") as f:
        f.write("\n".join(script_lines))

    return script_path


def execute_cell(
    cell: CodeCell,
    work_dir: Path,
    use_cache: bool = True,
    env_vars: Optional[Dict[str, str]] = None,
    force_rerun_cells: Optional[Set[str]] = None,
) -> ExecutionResult:
    """Execute a code cell using uv run."""
    import time

    deps_str = f" deps={','.join(cell.deps)}" if cell.deps else ""
    needs_str = f" needs={','.join(cell.needs)}" if cell.needs else ""
    print(f"cell={cell.id}{deps_str}{needs_str}")

    cache_key = generate_cache_key(cell, work_dir, env_vars)
    cache_dir = work_dir / ".uvnote" / "cache" / cache_key

    # Check if this cell should be forced to rerun
    force_rerun_this_cell = force_rerun_cells and cell.id in force_rerun_cells
    if force_rerun_this_cell:
        print(f"  force=dependencies")

    # Check cache
    if use_cache and cache_dir.exists() and not force_rerun_this_cell:
        result_file = cache_dir / "result.json"
        if result_file.exists():
            with open(result_file) as f:
                cached_result = json.load(f)
            # Integrity check using index/meta
            if not integrity_check(work_dir, cache_key):
                print(f"  cache=corrupt action=rerun")
            else:
                record_access(work_dir, cache_key)

                print(f"  cache=hit duration={cached_result['duration']:.2f}s")

                # Artifacts from index (fallback to dir scan)
                artifacts = get_artifacts(work_dir, cache_key)
                if not artifacts:
                    artifacts = []
                    if cache_dir.exists():
                        for item in cache_dir.iterdir():
                            if item.name not in {
                                "result.json",
                                "stdout.txt",
                                "stderr.txt",
                            }:
                                artifacts.append(str(item.relative_to(cache_dir)))

                if artifacts:
                    print(f"  artifacts={','.join(artifacts)}")
                print(f"  status=cached")

                return ExecutionResult(
                    cell_id=cell.id,
                    success=cached_result["success"],
                    stdout=cached_result["stdout"],
                    stderr=cached_result["stderr"],
                    duration=cached_result["duration"],
                    artifacts=artifacts,
                    cache_key=cache_key,
                )

    # Create execution directory
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"  cache=miss")

    # Create cell script
    script_path = create_cell_script(cell, work_dir)
    print(f"  script={script_path.relative_to(work_dir)}")

    # Execute with uv run (stateless temp directory)
    start_time = time.time()

    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)
        env_keys = list(env_vars.keys())
        print(f"  env_vars={','.join(env_keys)}")
    # Basic sandbox: minimal env
    env["PYTHONNOUSERSITE"] = "1"
    # Use ephemeral HOME inside temp dir for isolation
    tmp_home = None

    try:
        # Convert to absolute path for uv run
        absolute_script_path = script_path.resolve()
        print(f"  command=uv_run script={absolute_script_path.name}")

        with tempfile.TemporaryDirectory(prefix="uvnote-run-") as tmp:
            tmp_path = Path(tmp)
            tmp_home = tmp_path / "home"
            tmp_home.mkdir(parents=True, exist_ok=True)
            env["HOME"] = str(tmp_home)

            # Snapshot before
            before_set = set()
            # (Temp dir is empty, but keep logic for clarity)
            for root, dirs, files in os.walk(tmp_path):
                for fn in files:
                    before_set.add(str(Path(root) / fn))

            result = subprocess.run(
                ["uv", "run", str(absolute_script_path)],
                cwd=tmp_path,
                capture_output=True,
                text=True,
                env=env,
                timeout=300,  # 5 minute timeout
            )

            success = result.returncode == 0
            stdout = result.stdout
            stderr = result.stderr

            # Discover artifacts (prefer declared outputs)
            produced: List[Path] = []
            declared = set(cell.outputs or [])
            if declared:
                for rel in declared:
                    # Guard against traversal
                    rel_p = Path(rel)
                    if rel_p.is_absolute() or ".." in rel_p.parts:
                        continue
                    cp = tmp_path / rel_p
                    if cp.exists():
                        produced.append(cp)
            else:
                for root, dirs, files in os.walk(tmp_path):
                    for fn in files:
                        full = Path(root) / fn
                        # Ignore hidden files that tooling might create
                        if any(
                            part.startswith(".")
                            for part in full.relative_to(tmp_path).parts
                        ):
                            continue
                        produced.append(full)

            # Copy artifacts into cache dir (preserve relative paths)
            copied_rel: List[str] = []
            for p in produced:
                try:
                    rel = str(p.relative_to(tmp_path))
                except ValueError:
                    # Outside temp dir — skip
                    continue
                src = p
                dst = cache_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                # No symlink following
                try:
                    if src.is_file() and not src.is_symlink():
                        shutil.copy2(src, dst)
                        copied_rel.append(rel)
                except FileNotFoundError:
                    pass

    except subprocess.TimeoutExpired:
        print(f"  error=timeout duration=300s")
        success = False
        stdout = ""
        stderr = "Execution timed out after 300 seconds"
        copied_rel = []
    except Exception as e:
        print(f"  error={e}")
        success = False
        stdout = ""
        stderr = f"Execution failed: {e}"
        copied_rel = []

    duration = time.time() - start_time

    # Log completion
    print(f"  duration={duration:.2f}s")

    # Save outputs
    with open(cache_dir / "stdout.txt", "w") as f:
        f.write(stdout)

    with open(cache_dir / "stderr.txt", "w") as f:
        f.write(stderr)

    # Artifacts are the copied_rel list
    artifacts = copied_rel
    if artifacts:
        print(f"  artifacts={','.join(sorted(artifacts))}")

    print(f"  status={'success' if success else 'failed'}")

    # Save result metadata
    result_data = {
        "success": success,
        "stdout": stdout,
        "stderr": stderr,
        "duration": duration,
        "artifacts": artifacts,
    }

    with open(cache_dir / "result.json", "w") as f:
        json.dump(result_data, f, indent=2)

    # Update cache index and enforce cap
    try:
        init_db(work_dir)
        record_write(work_dir, cache_key, cache_dir, success, artifacts)
        cap = get_cache_cap_bytes()
        freed, removed = evict_to_target(work_dir, cap)
        if removed:
            print(f"  eviction=run freed_bytes={freed} removed={len(removed)}")
    except Exception as e:
        print(f"  cache_index_error={e}")

    return ExecutionResult(
        cell_id=cell.id,
        success=success,
        stdout=stdout,
        stderr=stderr,
        duration=duration,
        artifacts=artifacts,
        cache_key=cache_key,
    )


def execute_cells(
    cells: List[CodeCell],
    work_dir: Path,
    use_cache: bool = True,
    env_vars: Optional[Dict[str, str]] = None,
    force_rerun_cells: Optional[Set[str]] = None,
    incremental_callback: Optional[Callable[[List["ExecutionResult"]], None]] = None,
) -> List[ExecutionResult]:
    """Execute multiple cells in true topological dependency order.

    - Respects cell.needs (or depends alias) to order execution.
    - Injects env vars for upstream dependency cache dirs so downstream
      cells can discover artifacts at runtime.
    - Marks cells involved in cycles as failed with an explanatory error.
    """

    def sanitize_env_key(s: str) -> str:
        out = []
        for ch in s:
            if ch.isalnum():
                out.append(ch.upper())
            else:
                out.append("_")
        return "".join(out)

    def topo_sort(cells: List[CodeCell]) -> Tuple[List[str], Set[str]]:
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
        from collections import deque

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

    results: List[ExecutionResult] = []
    executed: Dict[str, ExecutionResult] = {}
    cells_by_id = {cell.id: cell for cell in cells}

    order, cyclic = topo_sort(cells)

    # Print dependency matrix
    print_dependency_matrix(cells)

    print(f"execution_plan:")
    print(f"  cells={len(order)}")
    print(f"  order={' -> '.join(order)}")
    if cyclic:
        print(f"  warning=cyclic_dependencies cells={','.join(cyclic)}")
    print()

    # Execute in computed order
    total_cells = len(order)
    for i, cid in enumerate(order, 1):
        print(f"progress={i}/{total_cells}")
        cell = cells_by_id[cid]
        # Build per-cell env vars including inputs from dependencies
        per_cell_env = dict(env_vars or {})
        if cell.needs:
            inputs_list: List[str] = []
            for need in cell.needs:
                if need in executed and executed[need].success:
                    dep_key = executed[need].cache_key
                    dep_dir = work_dir / ".uvnote" / "cache" / dep_key
                    env_key = f"UVNOTE_INPUT_{sanitize_env_key(need)}"
                    per_cell_env[env_key] = str(dep_dir)
                    inputs_list.append(env_key)
            if inputs_list:
                per_cell_env["UVNOTE_INPUTS"] = ",".join(inputs_list)

        result = execute_cell(
            cell, work_dir, use_cache, per_cell_env, force_rerun_cells
        )
        results.append(result)
        executed[cell.id] = result

        if result.success:
            print(f"  result=success")
        else:
            print(f"  result=failed stopping=true")
            break

        # Call incremental callback if provided
        if incremental_callback:
            incremental_callback(results)

    # Mark any cyclic cells as failed with an explanatory message
    for cid in cyclic:
        cell = cells_by_id[cid]
        print(f"cell={cell.id} status=skipped reason=cyclic_dependency")
        result = ExecutionResult(
            cell_id=cell.id,
            success=False,
            stdout="",
            stderr="Skipped: dependency cycle detected",
            duration=0.0,
            artifacts=[],
            cache_key=generate_cache_key(cell, work_dir, env_vars),
        )
        results.append(result)
        executed[cid] = result

    # Print execution summary
    successful = sum(1 for r in results if r.success)
    total = len(results)
    total_duration = sum(r.duration for r in results)

    print(f"execution_summary:")
    print(f"  success={successful}/{total}")
    print(f"  duration={total_duration:.2f}s")
    if successful < total:
        failed = [r.cell_id for r in results if not r.success]
        print(f"  failed={','.join(failed)}")
    print(f"  status={'complete' if successful == total else 'partial'}")

    return results
