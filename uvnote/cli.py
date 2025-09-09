"""CLI commands for uvnote."""

import shutil
import time
from pathlib import Path
from typing import Optional

import click
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from .executor import execute_cells
from .generator import generate_html
from .parser import parse_markdown, validate_cells


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
            if now - self.last_modified > 0.5:
                self.last_modified = now
                self.callback()


@click.group()
@click.version_option()
def main():
    """uvnote: Stateless, deterministic notebooks with uv and Markdown."""
    pass


@main.command()
@click.argument('file', type=click.Path(exists=True, path_type=Path))
@click.option('--output', '-o', type=click.Path(path_type=Path), help='Output directory (default: site/)')
@click.option('--no-cache', is_flag=True, help='Disable caching')
def build(file: Path, output: Optional[Path], no_cache: bool):
    """Build static HTML from markdown file."""
    
    if output is None:
        output = Path("site")
    
    work_dir = Path.cwd()
    
    # Read markdown file
    try:
        with open(file) as f:
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
    
    # Execute cells
    try:
        results = execute_cells(cells, work_dir, use_cache=not no_cache)
    except Exception as e:
        click.echo(f"Error executing cells: {e}", err=True)
        return 1
    
    # Check for failures
    failed_cells = [r for r in results if not r.success]
    if failed_cells:
        click.echo(f"Warning: {len(failed_cells)} cells failed execution")
        for result in failed_cells:
            click.echo(f"  - {result.cell_id}: {result.stderr.split()[0] if result.stderr else 'Unknown error'}")
    
    # Generate HTML
    output_file = output / f"{file.stem}.html"
    try:
        generate_html(content, config, cells, results, output_file, work_dir)
        click.echo(f"Generated: {output_file}")
    except Exception as e:
        click.echo(f"Error generating HTML: {e}", err=True)
        return 1
    
    return 0


@main.command()
@click.argument('file', type=click.Path(exists=True, path_type=Path))
@click.option('--cell', required=True, help='Cell ID to run')
@click.option('--no-cache', is_flag=True, help='Disable caching')
def run(file: Path, cell: str, no_cache: bool):
    """Run a single cell from markdown file."""
    
    work_dir = Path.cwd()
    
    # Read and parse markdown
    try:
        with open(file) as f:
            content = f.read()
        
        config, cells = parse_markdown(content)
        validate_cells(cells)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        return 1
    
    # Find the requested cell
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
    
    # Execute the cell
    from .executor import execute_cell
    
    try:
        result = execute_cell(target_cell, work_dir, use_cache=not no_cache)
    except Exception as e:
        click.echo(f"Error executing cell: {e}", err=True)
        return 1
    
    # Show results
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


@main.command()
@click.option('--all', is_flag=True, help='Clean all cache and site files')
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


@main.command()
@click.argument('file', type=click.Path(exists=True, path_type=Path))
@click.option('--output', '-o', type=click.Path(path_type=Path), help='Output directory (default: site/)')
@click.option('--host', default='localhost', help='Host to serve on')
@click.option('--port', default=8000, type=int, help='Port to serve on')
@click.option('--no-cache', is_flag=True, help='Disable caching')
def serve(file: Path, output: Optional[Path], host: str, port: int, no_cache: bool):
    """Watch markdown file, rebuild on changes, and serve HTML."""
    
    if output is None:
        output = Path("site")
    
    output_file = output / f"{file.stem}.html"
    
    def rebuild():
        """Rebuild the HTML file."""
        click.echo(f"Rebuilding {file}...")
        try:
            # Use click.Context to call build command
            ctx = click.Context(build)
            ctx.invoke(build, file=file, output=output, no_cache=no_cache)
            click.echo(f"Rebuilt: {output_file}")
        except Exception as e:
            click.echo(f"Rebuild failed: {e}", err=True)
    
    # Initial build
    rebuild()
    
    # Setup file watcher
    event_handler = MarkdownHandler(file, rebuild)
    observer = Observer()
    observer.schedule(event_handler, str(file.parent), recursive=False)
    observer.start()
    
    # Start simple HTTP server
    try:
        import http.server
        import socketserver
        import threading
        import webbrowser
        
        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(output), **kwargs)
        
        with socketserver.TCPServer((host, port), Handler) as httpd:
            url = f"http://{host}:{port}/{output_file.name}"
            click.echo(f"Serving at {url}")
            click.echo("Press Ctrl+C to stop")
            
            # Open browser
            threading.Timer(1.0, lambda: webbrowser.open(url)).start()
            
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                pass
    
    except Exception as e:
        click.echo(f"Server error: {e}", err=True)
    finally:
        observer.stop()
        observer.join()


if __name__ == '__main__':
    main()