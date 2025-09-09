"""Cell execution engine with uv run and caching."""

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

from .parser import CodeCell


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


def generate_cache_key(cell: CodeCell, env_vars: Optional[Dict[str, str]] = None) -> str:
    """Generate cache key for a cell based on code, dependencies, and environment."""
    content = {
        'code': cell.code,
        'deps': sorted(cell.deps),
        'env': sorted((env_vars or {}).items())
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
    
    with open(script_path, 'w') as f:
        f.write('\n'.join(script_lines))
    
    return script_path


def execute_cell(
    cell: CodeCell, 
    work_dir: Path, 
    use_cache: bool = True,
    env_vars: Optional[Dict[str, str]] = None
) -> ExecutionResult:
    """Execute a code cell using uv run."""
    import time
    
    cache_key = generate_cache_key(cell, env_vars)
    cache_dir = work_dir / ".uvnote" / "cache" / cache_key
    
    # Check cache
    if use_cache and cache_dir.exists():
        result_file = cache_dir / "result.json"
        if result_file.exists():
            with open(result_file) as f:
                cached_result = json.load(f)
            
            # Find artifacts
            artifacts = []
            if cache_dir.exists():
                for item in cache_dir.iterdir():
                    if item.name not in {'result.json', 'stdout.txt', 'stderr.txt'}:
                        artifacts.append(str(item.relative_to(cache_dir)))
            
            return ExecutionResult(
                cell_id=cell.id,
                success=cached_result['success'],
                stdout=cached_result['stdout'],
                stderr=cached_result['stderr'],
                duration=cached_result['duration'],
                artifacts=artifacts,
                cache_key=cache_key
            )
    
    # Create execution directory
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create cell script
    script_path = create_cell_script(cell, work_dir)
    
    # Execute with uv run
    start_time = time.time()
    
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)
    
    try:
        # Convert to absolute path for uv run
        absolute_script_path = script_path.resolve()
        result = subprocess.run(
            ["uv", "run", str(absolute_script_path)],
            cwd=cache_dir,
            capture_output=True,
            text=True,
            env=env,
            timeout=300  # 5 minute timeout
        )
        
        success = result.returncode == 0
        stdout = result.stdout
        stderr = result.stderr
        
    except subprocess.TimeoutExpired:
        success = False
        stdout = ""
        stderr = "Execution timed out after 300 seconds"
    except Exception as e:
        success = False
        stdout = ""
        stderr = f"Execution failed: {e}"
    
    duration = time.time() - start_time
    
    # Save outputs
    with open(cache_dir / "stdout.txt", "w") as f:
        f.write(stdout)
    
    with open(cache_dir / "stderr.txt", "w") as f:
        f.write(stderr)
    
    # Find artifacts (files created during execution)
    artifacts = []
    if cache_dir.exists():
        for item in cache_dir.iterdir():
            if item.name not in {'result.json', 'stdout.txt', 'stderr.txt'}:
                artifacts.append(str(item.relative_to(cache_dir)))
    
    # Save result metadata
    result_data = {
        'success': success,
        'stdout': stdout,
        'stderr': stderr,
        'duration': duration,
        'artifacts': artifacts
    }
    
    with open(cache_dir / "result.json", "w") as f:
        json.dump(result_data, f, indent=2)
    
    return ExecutionResult(
        cell_id=cell.id,
        success=success,
        stdout=stdout,
        stderr=stderr,
        duration=duration,
        artifacts=artifacts,
        cache_key=cache_key
    )


def execute_cells(
    cells: List[CodeCell], 
    work_dir: Path,
    use_cache: bool = True,
    env_vars: Optional[Dict[str, str]] = None
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
                out.append('_')
        return ''.join(out)

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

    # Execute in computed order
    for cid in order:
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

        result = execute_cell(cell, work_dir, use_cache, per_cell_env)
        results.append(result)
        executed[cell.id] = result
        if not result.success:
            print(f"Cell '{cell.id}' failed")

    # Mark any cyclic cells as failed with an explanatory message
    for cid in cyclic:
        cell = cells_by_id[cid]
        result = ExecutionResult(
            cell_id=cell.id,
            success=False,
            stdout="",
            stderr="Skipped: dependency cycle detected",
            duration=0.0,
            artifacts=[],
            cache_key=generate_cache_key(cell, env_vars)
        )
        results.append(result)
        executed[cid] = result

    return results
