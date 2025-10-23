"""File dependency resolution for cross-file cell dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .parser import CodeCell


@dataclass
class FileDependency:
    """Represents a dependency on another file or specific cell in that file."""

    file_path: Path  # Resolved absolute path to the markdown file
    cell_id: Optional[str] = (
        None  # If depending on specific cell, None means entire file
    )

    @property
    def key(self) -> str:
        """Unique key for this dependency."""
        if self.cell_id:
            return f"{self.file_path}:{self.cell_id}"
        return str(self.file_path)

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, other):
        if not isinstance(other, FileDependency):
            return False
        return self.key == other.key


def parse_file_reference(ref: str) -> Tuple[str, Optional[str]]:
    """
    Parse a file reference into file path and optional cell ID.

    Args:
        ref: File reference like "../math/algebra.md" or "../math/algebra.md:cellX"

    Returns:
        Tuple of (file_path, cell_id)

    Examples:
        "../math/algebra.md" -> ("../math/algebra.md", None)
        "../math/algebra.md:cellX" -> ("../math/algebra.md", "cellX")
        "math/algebra.md:cell_a:cell_b" -> ("math/algebra.md", "cell_a:cell_b")
    """
    if ":" in ref:
        # Split only on first colon to allow colons in cell IDs
        file_part, cell_part = ref.split(":", 1)
        return file_part.strip(), cell_part.strip()
    return ref.strip(), None


def resolve_file_dependencies(
    md_file: Path, cells: List[CodeCell], input_root: Path
) -> Dict[str, List[FileDependency]]:
    """
    Extract and resolve all file dependencies from cells in a markdown file.

    Args:
        md_file: Path to the markdown file being processed
        cells: List of code cells from the file
        input_root: Root directory for the build (used for validation)

    Returns:
        Dict mapping cell_id -> [FileDependency, ...]
        Only includes cells that have file dependencies

    Raises:
        ValueError: If a referenced file doesn't exist or path is invalid
    """
    cell_file_deps = {}

    for cell in cells:
        if not cell.needs_files:
            continue

        file_deps = []

        for ref in cell.needs_files:
            # Parse the reference
            file_part, cell_part = parse_file_reference(ref)

            # Resolve path relative to the current markdown file's directory
            dep_path = (md_file.parent / file_part).resolve()

            # Validate the file exists
            if not dep_path.exists():
                raise ValueError(
                    f"File dependency not found: '{ref}' "
                    f"(resolved to {dep_path}) "
                    f"referenced in {md_file.name}:{cell.id} at line {cell.line_start}"
                )

            # Validate it's a markdown file
            if not dep_path.suffix == ".md":
                raise ValueError(
                    f"File dependency must be a .md file: '{ref}' "
                    f"(resolved to {dep_path}) "
                    f"referenced in {md_file.name}:{cell.id}"
                )

            # Check for path traversal outside input_root (security)
            # Resolve input_root to absolute path for comparison
            input_root_resolved = input_root.resolve()
            try:
                dep_path.relative_to(input_root_resolved)
            except ValueError:
                raise ValueError(
                    f"File dependency outside project root: '{ref}' "
                    f"(resolved to {dep_path}, root is {input_root_resolved}) "
                    f"referenced in {md_file.name}:{cell.id}"
                )

            file_deps.append(FileDependency(dep_path, cell_part))

        if file_deps:
            cell_file_deps[cell.id] = file_deps

    return cell_file_deps


def build_file_dependency_graph(
    file_infos: Dict[Path, Tuple[List[CodeCell], Dict[str, List[FileDependency]]]],
) -> Dict[Path, Set[Path]]:
    """
    Build a file-level dependency graph from cell-level file dependencies.

    Args:
        file_infos: Dict mapping file_path -> (cells, cell_file_deps)
            where cell_file_deps is output from resolve_file_dependencies()

    Returns:
        Dict mapping file_path -> set of file_paths it depends on
        (i.e., adjacency list representation of the dependency graph)

    Example:
        {
            Path("programming/python.md"): {Path("math/algebra.md")},
            Path("math/algebra.md"): set(),
        }
    """
    graph = {}

    for file_path, (cells, cell_file_deps) in file_infos.items():
        # Collect all unique files this file depends on
        dep_files = set()

        for deps in cell_file_deps.values():
            for dep in deps:
                dep_files.add(dep.file_path)

        graph[file_path] = dep_files

    return graph


def topological_sort_files(
    graph: Dict[Path, Set[Path]],
) -> Tuple[List[Path], Set[Path]]:
    """
    Perform topological sort on file dependency graph using Kahn's algorithm.

    Args:
        graph: Adjacency list (file -> set of files it depends on)

    Returns:
        Tuple of (ordered_files, cyclic_files)
        - ordered_files: Files in dependency order (dependencies before dependents)
        - cyclic_files: Files involved in circular dependencies (can't be ordered)

    Example:
        Given: A depends on B, C depends on A
        graph = {A: {B}, B: set(), C: {A}}
        Returns: ([B, A, C], set())
    """
    from collections import deque

    # Get all files mentioned in the graph
    all_files = set(graph.keys())
    for deps in graph.values():
        all_files.update(deps)

    # Build reverse graph (who depends on me) and in-degree count
    in_degree = {f: 0 for f in all_files}
    reverse_graph = {f: [] for f in all_files}

    for file, deps in graph.items():
        for dep in deps:
            reverse_graph[dep].append(file)
            in_degree[file] += 1

    # Kahn's algorithm
    queue = deque([f for f in all_files if in_degree[f] == 0])
    ordered = []

    while queue:
        current = queue.popleft()
        ordered.append(current)

        # Reduce in-degree for files that depend on current
        for dependent in reverse_graph.get(current, []):
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    # Any files with non-zero in-degree are part of cycles
    cyclic = {f for f in all_files if in_degree[f] > 0}

    return ordered, cyclic


def detect_cycles(graph: Dict[Path, Set[Path]]) -> List[List[Path]]:
    """
    Detect all cycles in the file dependency graph.

    Args:
        graph: Adjacency list (file -> set of files it depends on)

    Returns:
        List of cycles, where each cycle is a list of files forming the cycle

    Example:
        If A->B->C->A, returns [[A, B, C, A]]
    """
    cycles = []
    visited = set()
    rec_stack = set()
    path = []

    def dfs(node: Path) -> bool:
        """DFS to detect cycles. Returns True if cycle found."""
        visited.add(node)
        rec_stack.add(node)
        path.append(node)

        for neighbor in graph.get(node, set()):
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in rec_stack:
                # Found a cycle
                cycle_start = path.index(neighbor)
                cycles.append(path[cycle_start:] + [neighbor])
                return True

        path.pop()
        rec_stack.remove(node)
        return False

    for node in graph.keys():
        if node not in visited:
            dfs(node)

    return cycles


def validate_cell_references(
    file_path: Path,
    cell_file_deps: Dict[str, List[FileDependency]],
    all_files_cells: Dict[Path, List[CodeCell]],
) -> None:
    """
    Validate that cell-specific file dependencies reference valid cells.

    Args:
        file_path: Path to the file being validated
        cell_file_deps: Cell file dependencies for this file
        all_files_cells: Dict mapping all file paths to their cells

    Raises:
        ValueError: If a cell reference is invalid
    """
    for cell_id, deps in cell_file_deps.items():
        for dep in deps:
            if dep.cell_id is None:
                # Whole file dependency, nothing to validate
                continue

            # Check if the referenced cell exists in the target file
            if dep.file_path not in all_files_cells:
                raise ValueError(
                    f"Internal error: File {dep.file_path} not in parsed files. "
                    f"Referenced by {file_path}:{cell_id}"
                )

            target_cells = all_files_cells[dep.file_path]
            target_cell_ids = {c.id for c in target_cells}

            if dep.cell_id not in target_cell_ids:
                available = (
                    ", ".join(sorted(target_cell_ids)) if target_cell_ids else "none"
                )
                raise ValueError(
                    f"Cell '{dep.cell_id}' not found in {dep.file_path.name}. "
                    f"Referenced by {file_path.name}:{cell_id} at line "
                    f"{next(c.line_start for c in all_files_cells[file_path] if c.id == cell_id)}. "
                    f"Available cells: {available}"
                )
