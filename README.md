# uvnote

> [!NOTE]
> uvnote is pre v1, so things are subject to change

<img width="2164" height="1392" alt="uvnotescreen" src="https://github.com/user-attachments/assets/5571a2a0-d849-4078-8395-436943d93082" />


`uvnote` is a "Stateless, deterministic notebooks with uv and Markdown."

In other words, its a alternative for Jupyter notebooks that is more lightweight, more reproducible, and more portable.

`uvnote` is kinda like a combination of a Markdown file and a Jupyter notebook and a static site generator.

## Concept

The premise is simple:

- you write normal markdown files with python code blocks
- each code block is expanded to a [uv/PEP 723 script](https://docs.astral.sh/uv/guides/scripts/#running-scripts)
- the output of each script is capture and rendered in the markdown file.
- all data/scripts are hashed and cached (in `.uvnote/cache`) so everything can be inspected and intelligently re-run when needed.
- no magic runtimes (relies on uv)
- no hidden state (cells are not stateful, they are just scripts)
- no special file formats (just plain markdown)

## How to use

Currently, the recommended way to use `uvnote` is to directly run the script from GitHub using `uvx`, this will download and run the latest version of `uvnote` without needing to install anything until we have a proper release.

```bash
uvx https://github.com/drbh/uvnote.git
```

outputs

```text
Usage: uvnote [OPTIONS] COMMAND [ARGS]...

  uvnote: Stateless, deterministic notebooks with uv and
  Markdown.

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  build  Build static HTML from markdown file.
  clean  Clear cache and site files.
  run    Run a single cell from markdown file.
  serve  Watch markdown file, rebuild on changes, and...
```

### Preview



If you're a `vscode` user, you can use the `uvnote-preview` extension to preview your uvnote files directly in VSCode.

https://github.com/drbh/uvnote-preview


https://github.com/user-attachments/assets/59a470e2-c3f6-46b7-b3ad-b4a0085b8dda


