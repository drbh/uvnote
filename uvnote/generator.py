"""Static HTML output generator."""

import html
import os
import platform
import shutil
from pathlib import Path
from typing import Dict, List

import markdown
from jinja2 import Environment, BaseLoader
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
    <script>
        // Apply theme immediately to prevent flicker
        (function() {
            const configTheme = '{{ config.theme }}';
            let theme;
            if (configTheme === 'auto') {
                theme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
            } else {
                theme = localStorage.getItem('uvnote-theme') || configTheme;
            }
            document.documentElement.setAttribute('data-theme', theme);
        })();
    </script>
    <style>
        :root[data-theme="light"] {
            --bg-primary: #ffffff;
            --bg-secondary: #f6f8fa;
            --bg-tertiary: #f8f9fa;
            --bg-code: #f8f9fa;
            --bg-error: #fdf2f2;
            --bg-artifact: #e6f3ff;
            --bg-artifact-hover: #d0e7ff;
            
            --text-primary: #333;
            --text-secondary: #656d76;
            --text-error: #c53030;
            --text-link: #0969da;
            
            --border-primary: #e1e5e9;
            --border-error: #e53e3e;
            --border-cell-failed: #d73a49;
            
            --shadow: rgba(0, 0, 0, 0.1);
        }
        
        :root[data-theme="dark"] {
            --bg-primary: #0a0a0a;
            --bg-secondary: #121212;
            --bg-tertiary: #181818;
            --bg-code: #0d0d0d;
            --bg-error: #1a0f0f;
            --bg-artifact: #151515;
            --bg-artifact-hover: #1a1a1a;
            
            --text-primary: #e0e0e0;
            --text-secondary: #888888;
            --text-error: #ff6b6b;
            --text-link: #64b5f6;
            
            --border-primary: #2a2a2a;
            --border-error: #ff6b6b;
            --border-cell-failed: #ff6b6b;
            
            --shadow: rgba(255, 255, 255, 0.05);
        }
        html {
            overscroll-behavior: none;
        }
        body {
            font-family: 'Cascadia Mono', 'Cascadia Code', 'JetBrains Mono', 'SF Mono', Monaco, 'Consolas', monospace;
            line-height: 1.4;
            max-width: 1000px;
            margin: 0 auto;
            padding: 15px;
            color: var(--text-primary);
            background: var(--bg-primary);
            transition: background-color 0.2s ease, color 0.2s ease;
            overscroll-behavior: none;
        }
        
        /* Two panel layout removed */
        
        .controls {
            position: fixed;
            top: 20px;
            right: 20px;
            display: flex;
            gap: 0.5rem;
            z-index: 1000;
        }
        
        .menu-button {
            position: relative;
            background: var(--bg-secondary);
            border: 1px solid var(--border-primary);
            padding: 8px 12px;
            border-radius: 4px;
            color: var(--text-secondary);
            cursor: pointer;
            font-family: inherit;
            font-size: 0.9rem;
            user-select: none;
        }
        
        .menu-button:hover {
            color: var(--text-primary);
            background: var(--bg-tertiary);
        }
        
        .menu-dropdown {
            position: absolute;
            top: 100%;
            right: 0;
            background: var(--bg-secondary);
            border: 1px solid var(--border-primary);
            border-radius: 4px;
            box-shadow: 0 4px 12px var(--shadow);
            min-width: 160px;
            opacity: 0;
            visibility: hidden;
            transform: translateY(-8px);
            transition: all 0.2s ease;
            z-index: 1001;
            margin-top: 4px;
        }
        
        .menu-button.active .menu-dropdown {
            opacity: 1;
            visibility: visible;
            transform: translateY(0);
        }
        
        .menu-item {
            display: block;
            padding: 8px 12px;
            color: var(--text-secondary);
            text-decoration: none;
            font-size: 0.85rem;
            border-bottom: 1px solid var(--border-primary);
            cursor: pointer;
        }
        
        .menu-item:last-child {
            border-bottom: none;
        }
        
        .menu-item:hover {
            background: var(--bg-tertiary);
            color: var(--text-primary);
        }
        
        .menu-checkbox {
            display: inline-block;
            width: 16px;
            font-family: monospace;
            color: var(--text-link);
        }
        
        .theme-toggle,
        .reset-toggle {
            background: var(--bg-secondary);
            border: 1px solid var(--border-primary);
            padding: 8px 12px;
            border-radius: 4px;
            color: var(--text-secondary);
            cursor: pointer;
            font-family: inherit;
            font-size: 0.9rem;
            user-select: none;
        }
        
        .theme-toggle:hover,
        .reset-toggle:hover {
            color: var(--text-primary);
            background: var(--bg-tertiary);
        }
        
        .system-info {
            background: var(--bg-secondary);
            border: 1px solid var(--border-primary);
            border-radius: 4px;
            padding: 8px 12px;
            margin-bottom: 16px;
            font-size: 0.85em;
            color: var(--text-secondary);
        }
        
        .system-info-header {
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 2px;
        }
        
        .system-info-content {
            font-family: monospace;
        }
        
        .theme-toggle, .reset-toggle {
            background: var(--bg-secondary);
            border: 1px solid var(--border-primary);
            border-radius: 2px;
            padding: 0.4rem 0.6rem;
            cursor: pointer;
            font-family: inherit;
            font-size: 0.8rem;
            color: var(--text-secondary);
            user-select: none;
            transition: all 0.2s ease;
            text-transform: lowercase;
            letter-spacing: 0;
        }
        
        .theme-toggle:hover, .reset-toggle:hover {
            background: var(--bg-tertiary);
            border-color: var(--text-secondary);
            color: var(--text-primary);
        }
        
        .minimap {
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 220px;
            max-height: 400px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-primary);
            border-radius: 2px;
            padding: 0.5rem;
            font-size: 0.7rem;
            overflow-y: auto;
            z-index: 100;
            opacity: 0.9;
            transition: opacity 0.2s ease;
        }
        
        .file-explorer {
            position: fixed;
            bottom: 20px; /* default; JS will stack */
            right: 20px;
            left: auto;
            top: auto;
            width: 220px;
            max-height: 400px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-primary);
            border-radius: 2px;
            padding: 0.5rem;
            font-size: 0.7rem;
            overflow-y: auto;
            z-index: 100;
            opacity: 0.9;
            transition: opacity 0.2s ease;
        }

        /* Drawing overlay */
        .draw-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            z-index: 80; /* under widgets (100) and controls (1000) */
            display: block;
            pointer-events: none; /* enabled only when a tool is active */
        }

        /* Tools widget */
        .tools-widget {
            position: fixed;
            bottom: 20px; /* default; JS will stack */
            right: 20px;
            left: auto;
            top: auto;
            width: 220px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-primary);
            border-radius: 2px;
            padding: 0.5rem;
            font-size: 0.7rem;
            z-index: 100;
            opacity: 0.95;
        }
        .tools-title {
            font-weight: bold;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
            padding-bottom: 0.25rem;
            border-bottom: 1px solid var(--border-primary);
            cursor: grab;
            user-select: none;
        }
        .tools-row { display: flex; gap: 0.4rem; flex-wrap: wrap; }
        .tool-button {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-primary);
            border-radius: 2px;
            padding: 0.25rem 0.4rem;
            cursor: pointer;
            color: var(--text-secondary);
            font-family: inherit;
            font-size: 0.75rem;
            user-select: none;
        }
        .tool-button:hover { color: var(--text-primary); }
        .tool-button.active { color: var(--text-primary); border-color: var(--text-secondary); background: var(--bg-secondary); }
        
        .minimap:hover, .file-explorer:hover {
            opacity: 1;
        }
        
        .minimap-title {
            font-weight: bold;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
            padding-bottom: 0.25rem;
            border-bottom: 1px solid var(--border-primary);
            cursor: grab; /* drag handle */
            user-select: none;
        }
        
        .minimap-item {
            display: block;
            color: var(--text-secondary);
            text-decoration: none;
            padding: 0.15rem 0;
            border-left: 2px solid transparent;
            padding-left: 0.5rem;
            transition: all 0.2s ease;
            cursor: pointer;
        }
        
        .minimap-item:hover {
            color: var(--text-primary);
            border-left-color: var(--text-secondary);
        }
        
        .minimap-item.active {
            color: var(--text-primary);
            border-left-color: var(--text-link);
        }
        
        .minimap-heading {
            font-weight: normal;
        }
        
        .minimap-heading.h1 { padding-left: 0.5rem; }
        .minimap-heading.h2 { padding-left: 1rem; }
        .minimap-heading.h3 { padding-left: 1.5rem; }
        .minimap-heading.h4 { padding-left: 2rem; }
        .minimap-heading.h5 { padding-left: 2.5rem; }
        .minimap-heading.h6 { padding-left: 3rem; }
        
        .minimap-cell {
            color: var(--text-link);
            opacity: 0.8;
            font-style: italic;
        }
        
        .minimap-cell:hover {
            opacity: 1;
        }
        
        .file-explorer-title {
            font-weight: bold;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
            padding-bottom: 0.25rem;
            border-bottom: 1px solid var(--border-primary);
            cursor: grab; /* drag handle */
            user-select: none;
        }
        
        .file-explorer-section {
            margin-bottom: 0.75rem;
        }
        
        .file-explorer-section-title {
            font-weight: bold;
            color: var(--text-secondary);
            font-size: 0.65rem;
            margin-bottom: 0.25rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .file-explorer-item {
            display: block;
            color: var(--text-secondary);
            text-decoration: none;
            padding: 0.1rem 0;
            margin-left: 0.5rem;
            transition: color 0.2s ease;
            cursor: pointer;
            font-family: monospace;
        }
        
        .file-explorer-item:hover {
            color: var(--text-primary);
        }
        
        .file-explorer-item.script {
            color: var(--text-link);
        }
        
        .file-explorer-item.artifact {
            color: var(--text-secondary);
            opacity: 0.8;
        }
        

        /* Hide widgets on smaller screens */
        @media (max-width: 768px) {
            .minimap, .file-explorer, .tools-widget {
                display: none;
            }
        }
        
        .cell {
            margin: 1rem 0;
            border: 1px solid var(--border-primary);
            border-radius: 2px;
            overflow: hidden;
            background: var(--bg-secondary);
        }
        .cell-header {
            background: var(--bg-secondary);
            padding: 0.5rem 1rem;
            border-bottom: 1px solid var(--border-primary);
            font-family: inherit;
            font-size: 0.85rem;
            color: var(--text-secondary);
            cursor: pointer;
            user-select: none;
            transition: background-color 0.2s ease;
        }
        .cell-header:hover {
            background: var(--bg-tertiary);
        }
        .collapse-indicators {
            color: var(--text-secondary);
            font-size: 0.8rem;
            opacity: 0.7;
        }
        .collapse-indicators span:hover {
            color: var(--text-primary);
            opacity: 1;
        }
        .cell-code {
            display: block;
            background: var(--bg-code);
        }
        .cell-code.collapsed {
            display: none;
        }
        .cell-code pre {
            margin: 0;
            padding: 0.75rem;
            background: var(--bg-code);
            overflow-x: auto;
            color: var(--text-primary);
        }
        .cell-output {
            padding: 0.75rem;
            background: var(--bg-primary);
        }
        .cell-output.collapsed {
            display: none;
        }
        .cell-stdout {
            background: var(--bg-tertiary);
            padding: 0.75rem;
            border-radius: 1px;
            margin: 0.25rem 0;
            font-family: inherit;
            font-size: 0.9rem;
            white-space: pre-wrap;
            color: var(--text-primary);
        }
        .cell-stderr {
            background: var(--bg-error);
            border-left: 2px solid var(--border-error);
            padding: 1rem;
            margin: 0.5rem 0;
            font-family: inherit;
            font-size: 0.9rem;
            color: var(--text-error);
            white-space: pre-wrap;
        }
        .uv-install-logs {
            margin: 0.5rem 0;
        }
        .uv-logs-header {
            cursor: pointer;
            padding: 0.75rem;
            border-left: 3px solid var(--border-color);
            font-family: inherit;
            font-size: 0.85rem;
            color: var(--text-secondary);
            user-select: none;
        }
        .uv-logs-content {
            background: var(--bg-secondary);
            padding: 1rem;
            border-left: 3px solid var(--border-color);
            white-space: pre-wrap;
            font-family: monospace;
            font-size: 0.85rem;
            color: var(--text-secondary);
            overflow-x: auto;
        }
        .cell-artifacts {
            margin: 1rem 0;
        }
        .cell-artifacts h4 {
            margin: 0 0 0.5rem 0;
            color: var(--text-secondary);
            font-size: 0.9rem;
        }
        .artifact {
            display: inline-block;
            background: var(--bg-artifact);
            padding: 0.25rem 0.5rem;
            border-radius: 1px;
            margin: 0.25rem 0.5rem 0.25rem 0;
            font-family: inherit;
            font-size: 0.8rem;
            color: var(--text-link);
            text-decoration: none;
            transition: background-color 0.2s ease;
            border: 1px solid var(--border-primary);
        }
        .artifact:hover {
            background: var(--bg-artifact-hover);
        }
        .artifact-preview {
            margin-top: 1rem;
        }
        .artifact-preview img {
            max-width: 100%;
            height: auto;
            border: 1px solid var(--border-primary);
            border-radius: 1px;
        }
        .artifact-preview svg {
            max-width: 100%;
            height: auto;
            border: 1px solid var(--border-primary);
            border-radius: 1px;
            display: block;
        }
        /* Style SVG text elements */
        .artifact-preview svg g {
            fill: var(--text-primary) !important;
        }
        /* Auto-theme SVG elements */
        .artifact-preview svg {
            background: transparent;
        }
        .cell-failed {
            border-color: var(--border-cell-failed);
        }
        .cell-failed .cell-header {
            background: var(--bg-error);
            color: var(--text-error);
        }
        .cell-commented {
            opacity: 0.6;
            border-style: dashed;
        }
        .cell-commented .cell-header {
            background: var(--bg-secondary);
            color: var(--text-secondary);
            font-style: italic;
        }
        .run-btn {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-primary);
            padding: 2px 6px;
            border-radius: 2px;
            color: var(--text-secondary);
            cursor: pointer;
            font-size: 0.75em;
            font-family: inherit;
            margin-left: 4px;
        }
        .run-btn:hover {
            color: var(--text-primary);
            background: var(--bg-primary);
        }
        .run-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        .copy-btn {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-primary);
            padding: 2px 6px;
            border-radius: 2px;
            color: var(--text-secondary);
            cursor: pointer;
            font-size: 0.75em;
            font-family: inherit;
            margin-left: 4px;
        }
        .copy-btn:hover {
            color: var(--text-primary);
            background: var(--bg-primary);
        }
        .copy-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        .raw-btn {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-primary);
            padding: 2px 6px;
            border-radius: 2px;
            color: var(--text-secondary);
            cursor: pointer;
            font-size: 0.75em;
            font-family: inherit;
            margin-left: 4px;
            text-decoration: none;
            display: inline-block;
        }
        .raw-btn:hover {
            color: var(--text-primary);
            background: var(--bg-primary);
            text-decoration: none;
        }
        .output-stale {
            opacity: 0.5;
            position: relative;
        }
        .output-stale::after {
            content: '⏳ updating...';
            position: absolute;
            top: 8px;
            right: 8px;
            background: var(--bg-secondary);
            padding: 4px 8px;
            border-radius: 2px;
            font-size: 0.75em;
            color: var(--text-secondary);
            border: 1px solid var(--border-primary);
        }
        h1, h2, h3, h4, h5, h6 {
            margin-top: 1.5rem;
            margin-bottom: 0.75rem;
            color: var(--text-primary);
        }
        h1 {
            margin-top: 0;
            margin-bottom: 1rem;
        }
        p {
            margin: 0.75rem 0;
            color: var(--text-primary);
        }
        a {
            color: var(--text-link);
        }
        img {
            max-width: 100%;
            height: auto;
            border-radius: 1px;
            box-shadow: none;
        }
        pre, code {
            font-family: 'Cascadia Mono', 'Cascadia Code', 'JetBrains Mono', 'SF Mono', Monaco, 'Consolas', monospace;
        }
        
        /* Line numbers */
        .highlight-with-lines {
            display: flex;
        }
        .line-numbers {
            background: var(--bg-tertiary);
            padding: 0.75rem 0.5rem;
            font-family: 'Cascadia Mono', 'Cascadia Code', 'JetBrains Mono', 'SF Mono', Monaco, 'Consolas', monospace;
            font-size: 0.9rem;
            color: var(--text-secondary);
            user-select: none;
            text-align: right;
            border-right: 1px solid var(--border-primary);
        }
        .line-numbers .line-number {
            display: block;
            line-height: 1.5;
        }
        .highlight-with-lines .highlight {
            flex: 1;
        }
        .highlight-with-lines .highlight pre {
            padding-left: 0.75rem;
        }
        
        /* Collapsed code styling */
        .cell-code.collapsed {
            display: none;
        }
        .cell-code.expanded {
            display: block;
        }
        {% if config.collapse_code %}
        .cell-code {
            display: none;
        }
        {% else %}
        .cell-code {
            display: block;
        }
        {% endif %}
        
        {{ pygments_css }}
        
        /* Custom CSS from frontmatter */
        {{ config.custom_css }}
        
        /* Cursor for tools */
        body[data-tool="arrow"] .main-content { 
            cursor: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="%23e53935" stroke-width="2"><path d="M12 19l7-7 3 3-7 7-3-3z"/><path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z"/><path d="M2 2l7.586 7.586"/><circle cx="11" cy="11" r="2"/></svg>') 12 12, crosshair;
        }
        body[data-tool="pen"] .main-content { 
            cursor: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="%23e53935" stroke-width="2"><path d="M12 19l7-7 3 3-7 7-3-3z"/><path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z"/><circle cx="4" cy="20" r="2" fill="%23e53935"/></svg>') 4 20, pointer;
        }
        body[data-tool="eraser"] .main-content { 
            cursor: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="%23e53935" stroke-width="2"><path d="M20 20H7l-7-7 7-7h13v14z"/><path d="M13 13l7-7"/><path d="M13 13L9 9"/></svg>') 12 12, auto;
        }

        /* Color picker styles */
        .tools-section-title {
            font-weight: bold;
            color: var(--text-secondary);
            font-size: 0.65rem;
            margin: 0.75rem 0 0.5rem 0;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .color-row {
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 0.25rem;
            margin-bottom: 0.5rem;
        }
        .color-swatch {
            width: 18px;
            height: 18px;
            border: 2px solid var(--border-primary);
            border-radius: 3px;
            cursor: pointer;
            transition: all 0.2s ease;
            position: relative;
        }
        .color-swatch:hover {
            transform: scale(1.1);
            border-color: var(--text-secondary);
        }
        .color-swatch.selected {
            border-color: var(--text-primary);
            box-shadow: 0 0 0 2px var(--text-link);
        }
        .color-swatch.selected::after {
            content: '✓';
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: white;
            font-size: 10px;
            font-weight: bold;
            text-shadow: 1px 1px 1px black;
        }
        .color-input {
            width: 24px;
            height: 24px;
            border: 2px solid var(--border-primary);
            border-radius: 3px;
            cursor: pointer;
            background: none;
            padding: 0;
            grid-column: span 2;
            justify-self: center;
        }
        .color-input:hover {
            border-color: var(--text-secondary);
        }
        
        /* Thickness slider styles */
        .thickness-row {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-top: 0.75rem;
        }
        .thickness-slider {
            flex: 1;
            -webkit-appearance: none;
            appearance: none;
            height: 4px;
            background: var(--border-primary);
            border-radius: 2px;
            outline: none;
            opacity: 0.7;
            transition: opacity 0.2s;
        }
        .thickness-slider:hover {
            opacity: 1;
        }
        .thickness-slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 12px;
            height: 12px;
            background: var(--text-link);
            border-radius: 50%;
            cursor: pointer;
        }
        .thickness-slider::-moz-range-thumb {
            width: 12px;
            height: 12px;
            background: var(--text-link);
            border-radius: 50%;
            cursor: pointer;
            border: none;
        }
        .thickness-value {
            font-size: 0.7rem;
            color: var(--text-secondary);
            min-width: 20px;
            text-align: right;
        }

        .highlight {
            background: none !important;
        }
        
        /* Loading animations */
        .loading-spinner {
            display: inline-block;
            width: 16px;
            height: 16px;
            border: 2px solid var(--border-primary);
            border-radius: 50%;
            border-top-color: var(--text-link);
            animation: spin 1s linear infinite;
            margin-right: 8px;
            vertical-align: middle;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .loading-skeleton {
            display: inline-block;
            background: var(--bg-tertiary);
            background: linear-gradient(
                90deg,
                var(--bg-tertiary) 25%,
                var(--bg-secondary) 50%,
                var(--bg-tertiary) 75%
            );
            background-size: 200% 100%;
            animation: loading-shimmer 2s ease-in-out infinite;
            border-radius: 2px;
            height: 1em;
            width: 80px;
            vertical-align: middle;
        }
        
        @keyframes loading-shimmer {
            0% { background-position: -200% 0; }
            100% { background-position: 200% 0; }
        }
        
        /* Loading state for cell output */
        .cell-output:has(.loading-spinner) {
            opacity: 0.7;
            background: var(--bg-secondary);
            border-left: 3px solid var(--text-link);
        }
    </style>
    <script>
        // --- Drag utilities ---
        function clamp(val, min, max) { return Math.max(min, Math.min(max, val)); }

        function restorePosition(el, storageKey) {
            try {
                const raw = localStorage.getItem(storageKey);
                if (!raw) return;
                const pos = JSON.parse(raw);
                if (typeof pos.left === 'number' && typeof pos.top === 'number') {
                    el.style.left = pos.left + 'px';
                    el.style.top = pos.top + 'px';
                    el.style.right = 'auto';
                    el.style.bottom = 'auto';
                }
            } catch (_) {}
        }

        function savePosition(el, storageKey) {
            try {
                const left = parseFloat(el.style.left || 'NaN');
                const top = parseFloat(el.style.top || 'NaN');
                if (!Number.isNaN(left) && !Number.isNaN(top)) {
                    localStorage.setItem(storageKey, JSON.stringify({ left, top }));
                }
            } catch (_) {}
        }


        function makeDraggable(el, storageKey, handleEl) {
            let dragging = false;
            let startX = 0, startY = 0; // cursor
            let origLeft = 0, origTop = 0; // element

            const onMove = (e) => {
                if (!dragging) return;
                const clientX = e.touches ? e.touches[0].clientX : e.clientX;
                const clientY = e.touches ? e.touches[0].clientY : e.clientY;
                const dx = clientX - startX;
                const dy = clientY - startY;
                const w = el.offsetWidth;
                const h = el.offsetHeight;
                const maxX = window.innerWidth - w;
                const maxY = window.innerHeight - h;
                const newLeft = clamp(origLeft + dx, 0, maxX);
                const newTop = clamp(origTop + dy, 0, maxY);
                el.style.left = newLeft + 'px';
                el.style.top = newTop + 'px';
                el.style.right = 'auto';
                el.style.bottom = 'auto';
            };

            const endDrag = () => {
                if (!dragging) return;
                dragging = false;
                document.removeEventListener('mousemove', onMove);
                document.removeEventListener('mouseup', endDrag);
                document.removeEventListener('touchmove', onMove);
                document.removeEventListener('touchend', endDrag);
                handleEl && (handleEl.style.cursor = 'grab');
                savePosition(el, storageKey);
                // ensure no-overlap constraint after a drag
                try { layoutWidgetsStackedBottomRight(); } catch (_) {}
            };

            const startDrag = (e) => {
                // Start from element's current on-screen rect
                const elRect = el.getBoundingClientRect();
                el.style.left = elRect.left + 'px';
                el.style.top = elRect.top + 'px';
                el.style.right = 'auto';
                el.style.bottom = 'auto';

                dragging = true;
                startX = e.touches ? e.touches[0].clientX : e.clientX;
                startY = e.touches ? e.touches[0].clientY : e.clientY;
                origLeft = elRect.left;
                origTop = elRect.top;

                document.addEventListener('mousemove', onMove);
                document.addEventListener('mouseup', endDrag);
                document.addEventListener('touchmove', onMove, { passive: false });
                document.addEventListener('touchend', endDrag);
                handleEl && (handleEl.style.cursor = 'grabbing');
                e.preventDefault();
            };

            (handleEl || el).addEventListener('mousedown', startDrag);
            (handleEl || el).addEventListener('touchstart', startDrag, { passive: false });

            // Apply any saved position on init
            restorePosition(el, storageKey);
        }
        function toggleCell(cellId) {
            const codeElement = document.getElementById('code-' + cellId);
            const outputElement = document.getElementById('output-' + cellId);
            
            if (codeElement) {
                codeElement.classList.toggle('collapsed');
            }
            if (outputElement) {
                outputElement.classList.toggle('collapsed');
            }
            
            updateIndicators(cellId);
        }
        
        function toggleCode(cellId) {
            const codeElement = document.getElementById('code-' + cellId);
            if (codeElement) {
                codeElement.classList.toggle('collapsed');
                updateIndicators(cellId);
            }
        }
        
        function toggleOutput(cellId) {
            const outputElement = document.getElementById('output-' + cellId);
            if (outputElement) {
                outputElement.classList.toggle('collapsed');
                updateIndicators(cellId);
            }
        }

        function toggleUvLogs(headerElement) {
            const contentElement = headerElement.nextElementSibling;
            if (contentElement) {
                const isCollapsed = contentElement.style.display === 'none';
                contentElement.style.display = isCollapsed ? 'block' : 'none';
                headerElement.textContent = isCollapsed ? '▼ UV Install Logs' : '▶ UV Install Logs';

                // Update the header indicator if it exists
                const uvLogsDiv = headerElement.parentElement;
                if (uvLogsDiv && uvLogsDiv.id && uvLogsDiv.id.startsWith('uv-logs-')) {
                    const cellId = uvLogsDiv.id.replace('uv-logs-', '');
                    const indicatorElement = document.getElementById('uv-indicator-' + cellId);
                    if (indicatorElement) {
                        indicatorElement.textContent = isCollapsed ? '▼ uv-logs' : '▶ uv-logs';
                    }
                }
            }
        }

        function toggleUvLogsFromHeader(cellId) {
            const uvLogsElement = document.getElementById('uv-logs-' + cellId);
            const indicatorElement = document.getElementById('uv-indicator-' + cellId);
            if (uvLogsElement) {
                const headerElement = uvLogsElement.querySelector('.uv-logs-header');
                const contentElement = uvLogsElement.querySelector('.uv-logs-content');
                if (contentElement && headerElement) {
                    const isCollapsed = contentElement.style.display === 'none';
                    contentElement.style.display = isCollapsed ? 'block' : 'none';
                    headerElement.textContent = isCollapsed ? '▼ UV Install Logs' : '▶ UV Install Logs';
                    if (indicatorElement) {
                        indicatorElement.textContent = isCollapsed ? '▼ uv-logs' : '▶ uv-logs';
                    }
                }
            }
        }
        
        function updateIndicators(cellId) {
            const codeElement = document.getElementById('code-' + cellId);
            const outputElement = document.getElementById('output-' + cellId);
            const indicators = document.querySelector(`[onclick*="${cellId}"]`)?.closest('.cell-header')?.querySelector('.collapse-indicators');
            
            if (indicators) {
                const codeCollapsed = codeElement && codeElement.classList.contains('collapsed');
                const outputCollapsed = outputElement && outputElement.classList.contains('collapsed');
                
                const codeIcon = codeCollapsed ? '▶' : '▼';
                const outputIcon = outputCollapsed ? '▶' : '▼';
                
                const codeSpan = indicators.querySelector('[onclick*="toggleCode"]');
                const outputSpan = indicators.querySelector('[onclick*="toggleOutput"]');
                
                if (codeSpan) codeSpan.innerHTML = `${codeIcon} code`;
                if (outputSpan) outputSpan.innerHTML = `${outputIcon} output`;
            }
        }
        
        function toggleTheme() {
            const html = document.documentElement;
            const currentTheme = html.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            html.setAttribute('data-theme', newTheme);
            localStorage.setItem('uvnote-theme', newTheme);
            updateThemeIcon();
        }
        
        // Two panel code removed
        
        function updateThemeIcon() {
            const theme = document.documentElement.getAttribute('data-theme');
            const toggle = document.querySelector('.theme-toggle');
            if (toggle) {
                toggle.textContent = theme === 'dark' ? 'light' : 'dark';
            }
        }

        function resetLayout() {
            try {
                // Clear all uvnote-* keys
                const allKeys = Object.keys(localStorage);
                const uvnoteKeys = allKeys.filter(key => key.startsWith('uvnote-'));
                uvnoteKeys.forEach(k => localStorage.removeItem(k));
            } catch (_) {}
            // Reload to reinitialize UI with defaults
            location.reload();
        }

        function toggleMenu() {
            const menuButton = document.querySelector('.menu-button');
            if (menuButton) {
                menuButton.classList.toggle('active');
            }
        }

        function toggleWidget(widgetName) {
            let widget;
            let checkbox;
            
            // Close the menu first
            const menuButton = document.querySelector('.menu-button');
            if (menuButton) {
                menuButton.classList.remove('active');
            }
            
            switch(widgetName) {
                case 'tools':
                    widget = document.querySelector('.tools-widget');
                    checkbox = document.getElementById('checkbox-tools');
                    break;
                case 'file-explorer':
                    widget = document.querySelector('.file-explorer');
                    checkbox = document.getElementById('checkbox-file-explorer');
                    break;
                case 'minimap':
                    widget = document.querySelector('.minimap');
                    checkbox = document.getElementById('checkbox-minimap');
                    break;
                default:
                    return;
            }
            
            if (widget && checkbox) {
                const isVisible = getComputedStyle(widget).display !== 'none';
                widget.style.display = isVisible ? 'none' : 'block';
                checkbox.textContent = isVisible ? '☐' : '☑';
                
                // Save state to localStorage
                try {
                    localStorage.setItem(`uvnote-widget-${widgetName}`, isVisible ? 'hidden' : 'visible');
                } catch (_) {}
                
                // Re-layout widgets after visibility change
                try { 
                    layoutWidgetsStackedBottomRight(); 
                } catch (_) {}
            }
        }
        
        function initializeWidgetVisibility() {
            const widgets = [
                { name: 'tools', selector: '.tools-widget' },
                { name: 'file-explorer', selector: '.file-explorer' },
                { name: 'minimap', selector: '.minimap' }
            ];
            
            widgets.forEach(({ name, selector }) => {
                const savedState = localStorage.getItem(`uvnote-widget-${name}`) || 'hidden';
                const widget = document.querySelector(selector);
                const checkbox = document.getElementById(`checkbox-${name}`);
                
                if (widget && checkbox) {
                    const isVisible = savedState === 'visible';
                    widget.style.display = isVisible ? 'block' : 'none';
                    checkbox.textContent = isVisible ? '☑' : '☐';
                }
            });
        }

        // Close menu when clicking outside
        document.addEventListener('click', function(event) {
            const menuButton = document.querySelector('.menu-button');
            // Don't close if clicking on a menu item (let the item handler close it)
            if (menuButton && !menuButton.contains(event.target)) {
                menuButton.classList.remove('active');
            }
        });

        // Layout: stack widgets bottom-right and equalize widths
        function hasCustomWidgetPositions() {
            try {
                return (
                    localStorage.getItem('uvnote-minimap-pos') ||
                    localStorage.getItem('uvnote-file-explorer-pos') ||
                    localStorage.getItem('uvnote-tools-pos')
                );
            } catch (_) { return false; }
        }

        function rectsOverlap(r1, r2) {
            return !(r1.right <= r2.left || r2.right <= r1.left || r1.bottom <= r2.top || r2.bottom <= r1.top);
        }

        function widgetsOverlap(widgets) {
            for (let i = 0; i < widgets.length; i++) {
                const a = widgets[i];
                const ra = a.getBoundingClientRect();
                for (let j = i + 1; j < widgets.length; j++) {
                    const b = widgets[j];
                    const rb = b.getBoundingClientRect();
                    if (rectsOverlap(ra, rb)) return true;
                }
            }
            return false;
        }

        function applyStackLayout(widgets, order) {
            if (!widgets.length) return;
            // Fixed equal width
            const fixedWidth = 220;
            widgets.forEach(el => { el.style.width = fixedWidth + 'px'; });

            // Fit heights if needed to avoid overflow
            const gap = 12;
            const available = Math.max(0, window.innerHeight - 40 - gap * (order.length - 1));
            const eachMax = Math.floor(available / order.length);
            order.forEach(el => {
                el.style.maxHeight = eachMax + 'px';
                el.style.overflowY = 'auto';
            });

            // Stack bottom-up in the requested order
            let bottomOffset = 20; // base gutter
            order.forEach(el => {
                el.style.left = 'auto';
                el.style.top = 'auto';
                el.style.right = '20px';
                el.style.bottom = bottomOffset + 'px';
                bottomOffset += el.offsetHeight + gap;
            });
        }

        function layoutWidgetsStackedBottomRight() {
            const minimap = document.querySelector('.minimap');
            const fileExplorer = document.querySelector('.file-explorer');
            const tools = document.querySelector('.tools-widget');
            const widgets = [minimap, fileExplorer, tools].filter(el => el && getComputedStyle(el).display !== 'none');
            if (!widgets.length) return;

            const order = [minimap, fileExplorer, tools].filter(Boolean).filter(el => getComputedStyle(el).display !== 'none');

            // If user placed custom positions and there is no overlap, respect them.
            if (hasCustomWidgetPositions() && !widgetsOverlap(widgets)) return;

            applyStackLayout(widgets, order);
        }
        
        // Panel icon removed
        
        let _minimapScrollContainer = null;
        let _minimapScrollHandler = null;
        function initMinimap() {
            // Generate minimap content
            const minimap = createMinimap();
            document.body.appendChild(minimap);
            // Make draggable (use title as handle)
            const mTitle = minimap.querySelector('.minimap-title');
            makeDraggable(minimap, 'uvnote-minimap-pos', mTitle);

            // Attach scroll listener to window (two-panel removed)
            _minimapScrollContainer = window;

            if (_minimapScrollContainer) {
                _minimapScrollHandler = () => updateMinimapActive();
                if (_minimapScrollContainer === window) {
                    window.addEventListener('scroll', _minimapScrollHandler);
                } else {
                    _minimapScrollContainer.addEventListener('scroll', _minimapScrollHandler);
                }
            }
            updateMinimapActive();
        }

        function teardownMinimap() {
            const minimap = document.querySelector('.minimap');
            if (minimap && minimap.parentNode) minimap.parentNode.removeChild(minimap);
            if (_minimapScrollContainer && _minimapScrollHandler) {
                if (_minimapScrollContainer === window) {
                    window.removeEventListener('scroll', _minimapScrollHandler);
                } else {
                    _minimapScrollContainer.removeEventListener('scroll', _minimapScrollHandler);
                }
            }
            _minimapScrollContainer = null;
            _minimapScrollHandler = null;
        }
        
        function initFileExplorer() {
            // Generate file explorer content
            const fileExplorer = createFileExplorer();
            document.body.appendChild(fileExplorer);
        }
        
        function createMinimap() {
            const minimap = document.createElement('div');
            minimap.className = 'minimap';
            
            const title = document.createElement('div');
            title.className = 'minimap-title';
            title.textContent = 'navigation';
            minimap.appendChild(title);
            
            // Find all headings and cells
            const root = document.querySelector('.main-content') || document;
            const headings = root.querySelectorAll('h1, h2, h3, h4, h5, h6');
            const cells = root.querySelectorAll('.cell');
            
            // Combine and sort by position
            const items = [];
            
            headings.forEach(heading => {
                const id = heading.id || generateId(heading.textContent);
                if (!heading.id) heading.id = id;
                
                items.push({
                    element: heading,
                    type: 'heading',
                    level: parseInt(heading.tagName.charAt(1)),
                    text: heading.textContent.trim(),
                    id: id,
                    position: heading.getBoundingClientRect().top + window.scrollY
                });
            });
            
            cells.forEach(cell => {
                const header = cell.querySelector('.cell-header');
                if (header) {
                    const id = cell.id || `cell-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
                    if (!cell.id) cell.id = id;
                    
                    items.push({
                        element: cell,
                        type: 'cell',
                        text: header.textContent.trim(),
                        id: id,
                        position: cell.getBoundingClientRect().top + window.scrollY
                    });
                }
            });
            
            // Sort by position
            items.sort((a, b) => a.position - b.position);
            
            // Create minimap items
            items.forEach(item => {
                const link = document.createElement('a');
                link.className = `minimap-item ${item.type === 'heading' ? 'minimap-heading' : 'minimap-cell'}`;
                if (item.type === 'heading') {
                    link.classList.add(`h${item.level}`);
                }
                link.textContent = item.text.length > 25 ? item.text.substring(0, 22) + '...' : item.text;
                link.href = `#${item.id}`;
                link.onclick = function(e) {
                    e.preventDefault();
                    item.element.scrollIntoView({ behavior: 'smooth', block: 'start' });
                };
                minimap.appendChild(link);
            });
            
            return minimap;
        }
        
        function generateId(text) {
            return text.toLowerCase()
                .replace(/[^a-z0-9]+/g, '-')
                .replace(/^-+|-+$/g, '')
                .substring(0, 20);
        }
        
        function updateMinimapActive() {
            const minimapItems = document.querySelectorAll('.minimap-item');
            const container = _minimapScrollContainer || window;
            const containerRect = container === window ? null : container.getBoundingClientRect();
            const scrollPos = (container === window ? window.scrollY : container.scrollTop) + 100; // Offset for better detection
            
            let activeItem = null;
            minimapItems.forEach(item => {
                const targetId = item.getAttribute('href').substring(1);
                const target = document.getElementById(targetId);
                
                if (target) {
                    const rectTop = target.getBoundingClientRect().top;
                    const targetPos = (container === window)
                        ? rectTop + window.scrollY
                        : rectTop - containerRect.top + container.scrollTop;
                    if (targetPos <= scrollPos) {
                        activeItem = item;
                    }
                }
                
                item.classList.remove('active');
            });
            
            if (activeItem) {
                activeItem.classList.add('active');
            }
        }
        
        function createFileExplorer() {
            const fileExplorer = document.createElement('div');
            fileExplorer.className = 'file-explorer';
            
            const title = document.createElement('div');
            title.className = 'file-explorer-title';
            title.textContent = 'files';
            fileExplorer.appendChild(title);
            // Make draggable (use title as handle)
            makeDraggable(fileExplorer, 'uvnote-file-explorer-pos', title);
            
            // Scripts section
            const scriptsSection = document.createElement('div');
            scriptsSection.className = 'file-explorer-section';
            
            const scriptsTitle = document.createElement('div');
            scriptsTitle.className = 'file-explorer-section-title';
            scriptsTitle.textContent = 'scripts';
            scriptsSection.appendChild(scriptsTitle);
            
            // Find all cells and list their script files (single panel)
            const root = document.querySelector('.main-content') || document;
            const cells = root.querySelectorAll('.cell');
            cells.forEach(cell => {
                const header = cell.querySelector('.cell-header');
                if (header) {
                    const cellText = header.textContent.trim();
                    const cellMatch = cellText.match(/Cell: ([a-zA-Z_][a-zA-Z0-9_]*)/);
                    if (cellMatch) {
                        const cellId = cellMatch[1];
                        const scriptItem = document.createElement('div');
                        scriptItem.className = 'file-explorer-item script';
                        scriptItem.textContent = `${cellId}.py`;
                        scriptItem.onclick = function() {
                            cell.scrollIntoView({ behavior: 'smooth', block: 'start' });
                        };
                        scriptsSection.appendChild(scriptItem);
                    }
                }
            });
            
            fileExplorer.appendChild(scriptsSection);
            
            // Artifacts section
            const artifactsSection = document.createElement('div');
            artifactsSection.className = 'file-explorer-section';
            
            const artifactsTitle = document.createElement('div');
            artifactsTitle.className = 'file-explorer-section-title';
            artifactsTitle.textContent = 'artifacts';
            artifactsSection.appendChild(artifactsTitle);
            
            // Find all artifact links (single panel)
            const artifactsRoot = document.querySelector('.main-content') || document;
            const artifacts = artifactsRoot.querySelectorAll('.artifact');
            if (artifacts.length === 0) {
                const noArtifacts = document.createElement('div');
                noArtifacts.className = 'file-explorer-item artifact';
                noArtifacts.textContent = '(none)';
                noArtifacts.style.opacity = '0.5';
                artifactsSection.appendChild(noArtifacts);
            } else {
                artifacts.forEach(artifact => {
                    const artifactItem = document.createElement('div');
                    artifactItem.className = 'file-explorer-item artifact';
                    artifactItem.textContent = artifact.textContent;
                    artifactItem.onclick = function() {
                        artifact.click();
                    };
                    artifactsSection.appendChild(artifactItem);
                });
            }
            
            fileExplorer.appendChild(artifactsSection);
            
            return fileExplorer;
        }

        // Tools widget
        let _cursorX = 0;
        let _cursorY = 0;
        let _cursorVisible = false;
        
        function setActiveTool(tool) {
            if (!tool || tool === 'none') {
                document.body.dataset.tool = 'none';
                localStorage.setItem('uvnote-active-tool', 'none');
                setOverlayActive(false);
                _cursorVisible = false;
                // Remove active class from all tool buttons when deactivating
                const toolButtons = document.querySelectorAll('.tools-widget .tool-button');
                toolButtons.forEach(btn => btn.classList.remove('active'));
                return;
            }
            document.body.dataset.tool = tool;
            localStorage.setItem('uvnote-active-tool', tool);
            setOverlayActive(true);
            _cursorVisible = true;
        }

        // Make setActiveTool globally accessible for ESC key handler
        window.setActiveTool = setActiveTool;

        function getArrowColor() {
            const saved = localStorage.getItem('uvnote-arrow-color');
            if (saved) return saved;
            return '#e53935'; // Default red color
        }

        function setStoredArrowColor(color) {
            try { localStorage.setItem('uvnote-arrow-color', color); } catch (_) {}
        }

        function getLineThickness() {
            const saved = localStorage.getItem('uvnote-line-thickness');
            if (saved) return parseInt(saved, 10);
            return 4; // default thickness
        }

        function setStoredLineThickness(thickness) {
            try { localStorage.setItem('uvnote-line-thickness', thickness); } catch (_) {}
        }

        function getFadeoutTime() {
            const saved = localStorage.getItem('uvnote-fadeout-time');
            if (saved) return parseInt(saved, 10);
            return 5; // default 5 seconds
        }

        function setStoredFadeoutTime(seconds) {
            try { localStorage.setItem('uvnote-fadeout-time', seconds); } catch (_) {}
        }

        function createToolsWidget() {
            const tools = document.createElement('div');
            tools.className = 'tools-widget';

            const title = document.createElement('div');
            title.className = 'tools-title';
            title.textContent = 'tools';
            tools.appendChild(title);

            const row = document.createElement('div');
            row.className = 'tools-row';
            tools.appendChild(row);

            // Arrow tool
            const arrowBtn = document.createElement('div');
            arrowBtn.className = 'tool-button';
            arrowBtn.textContent = 'arrow';
            arrowBtn.onclick = function() {
                const isActive = arrowBtn.classList.contains('active');
                if (isActive) {
                    arrowBtn.classList.remove('active');
                    setActiveTool('none');
                } else {
                    tools.querySelectorAll('.tool-button').forEach(b => b.classList.remove('active'));
                    arrowBtn.classList.add('active');
                    setActiveTool('arrow');
                }
            };
            row.appendChild(arrowBtn);

            // Pen tool
            const penBtn = document.createElement('div');
            penBtn.className = 'tool-button';
            penBtn.textContent = 'pen';
            penBtn.onclick = function() {
                const isActive = penBtn.classList.contains('active');
                if (isActive) {
                    penBtn.classList.remove('active');
                    setActiveTool('none');
                } else {
                    tools.querySelectorAll('.tool-button').forEach(b => b.classList.remove('active'));
                    penBtn.classList.add('active');
                    setActiveTool('pen');
                }
            };
            row.appendChild(penBtn);

            // Eraser tool
            const eraseBtn = document.createElement('div');
            eraseBtn.className = 'tool-button';
            eraseBtn.textContent = 'eraser';
            eraseBtn.onclick = function() {
                const isActive = eraseBtn.classList.contains('active');
                if (isActive) {
                    eraseBtn.classList.remove('active');
                    setActiveTool('none');
                } else {
                    tools.querySelectorAll('.tool-button').forEach(b => b.classList.remove('active'));
                    eraseBtn.classList.add('active');
                    setActiveTool('eraser');
                }
            };
            row.appendChild(eraseBtn);

            // Spotlight tool
            const spotlightBtn = document.createElement('div');
            spotlightBtn.className = 'tool-button';
            spotlightBtn.textContent = 'spotlight';
            spotlightBtn.onclick = function() {
                const isActive = spotlightBtn.classList.contains('active');
                if (isActive) {
                    spotlightBtn.classList.remove('active');
                    setActiveTool('none');
                } else {
                    tools.querySelectorAll('.tool-button').forEach(b => b.classList.remove('active'));
                    spotlightBtn.classList.add('active');
                    setActiveTool('spotlight');
                }
            };
            row.appendChild(spotlightBtn);

            // Clear all
            const clearBtn = document.createElement('div');
            clearBtn.className = 'tool-button';
            clearBtn.textContent = 'clear';
            clearBtn.onclick = function() {
                _shapes = [];
                saveShapes();
                renderOverlay();
            };
            row.appendChild(clearBtn);

            // Restore active state from storage
            const saved = localStorage.getItem('uvnote-active-tool') || 'none';
            if (saved === 'arrow') {
                arrowBtn.classList.add('active');
                setActiveTool('arrow');
            } else if (saved === 'pen') {
                penBtn.classList.add('active');
                setActiveTool('pen');
            } else if (saved === 'eraser') {
                eraseBtn.classList.add('active');
                setActiveTool('eraser');
            } else if (saved === 'spotlight') {
                spotlightBtn.classList.add('active');
                setActiveTool('spotlight');
            }

            // Color selector
            const colorTitle = document.createElement('div');
            colorTitle.className = 'tools-section-title';
            colorTitle.textContent = 'color';
            tools.appendChild(colorTitle);

            const colorRow = document.createElement('div');
            colorRow.className = 'tools-row color-row';
            tools.appendChild(colorRow);

            const swatchColors = [
                // Primary colors
                '#e53935', '#fb8c00', '#fdd835', '#43a047', '#1e88e5', '#8e24aa',
                // Additional useful colors  
                '#ff5722', '#795548', '#607d8b', '#9c27b0',
                // Grayscale
                '#000000', '#424242', '#9e9e9e', '#ffffff'
            ];
            const swatches = [];
            swatchColors.forEach(c => {
                const s = document.createElement('div');
                s.className = 'color-swatch';
                s.style.backgroundColor = c;
                s.title = c;
                s.onclick = () => {
                    setStoredArrowColor(c);
                    refreshColorUI(c);
                    if (_cursorVisible) renderOverlay();
                };
                colorRow.appendChild(s);
                swatches.push(s);
            });

            const colorInput = document.createElement('input');
            colorInput.type = 'color';
            colorInput.className = 'color-input';
            colorInput.oninput = () => {
                setStoredArrowColor(colorInput.value);
                refreshColorUI(colorInput.value);
                if (_cursorVisible) renderOverlay();
            };
            colorRow.appendChild(colorInput);

            function refreshColorUI(selected) {
                const selectedHex = selected.startsWith('#') ? selected.toLowerCase() : rgbToHex(selected);
                
                swatches.forEach((s, i) => {
                    const swatchHex = swatchColors[i].toLowerCase();
                    if (swatchHex === selectedHex) {
                        s.classList.add('selected');
                    } else {
                        s.classList.remove('selected');
                    }
                });
                
                try { 
                    colorInput.value = selectedHex; 
                } catch (_) {}
            }

            function rgbToHex(rgb) {
                const m = rgb.match(/rgba?\\((\\d+),\\s*(\\d+),\\s*(\\d+)/i);
                if (!m) return '#000000';
                const r = parseInt(m[1]).toString(16).padStart(2, '0');
                const g = parseInt(m[2]).toString(16).padStart(2, '0');
                const b = parseInt(m[3]).toString(16).padStart(2, '0');
                return `#${r}${g}${b}`;
            }

            // Restore color selection
            refreshColorUI(getArrowColor());

            // Thickness slider
            const thicknessTitle = document.createElement('div');
            thicknessTitle.className = 'tools-section-title';
            thicknessTitle.textContent = 'thickness';
            tools.appendChild(thicknessTitle);

            const thicknessRow = document.createElement('div');
            thicknessRow.className = 'thickness-row';
            tools.appendChild(thicknessRow);

            const thicknessSlider = document.createElement('input');
            thicknessSlider.type = 'range';
            thicknessSlider.className = 'thickness-slider';
            thicknessSlider.min = '1';
            thicknessSlider.max = '10';
            thicknessSlider.value = getLineThickness();
            
            const thicknessValue = document.createElement('span');
            thicknessValue.className = 'thickness-value';
            thicknessValue.textContent = thicknessSlider.value + 'px';

            thicknessSlider.oninput = function() {
                const value = parseInt(thicknessSlider.value, 10);
                setStoredLineThickness(value);
                thicknessValue.textContent = value + 'px';
                if (_cursorVisible) renderOverlay();
            };

            thicknessRow.appendChild(thicknessSlider);
            thicknessRow.appendChild(thicknessValue);

            // Fadeout time slider
            const fadeoutTitle = document.createElement('div');
            fadeoutTitle.className = 'tools-section-title';
            fadeoutTitle.textContent = 'fadeout time';
            tools.appendChild(fadeoutTitle);

            const fadeoutRow = document.createElement('div');
            fadeoutRow.className = 'thickness-row';
            tools.appendChild(fadeoutRow);

            const fadeoutSlider = document.createElement('input');
            fadeoutSlider.type = 'range';
            fadeoutSlider.className = 'thickness-slider';
            fadeoutSlider.min = '0';
            fadeoutSlider.max = '30';
            fadeoutSlider.value = getFadeoutTime();
            
            const fadeoutValue = document.createElement('span');
            fadeoutValue.className = 'thickness-value';
            fadeoutValue.textContent = fadeoutSlider.value === '0' ? 'never' : fadeoutSlider.value + 's';

            fadeoutSlider.oninput = function() {
                const value = parseInt(fadeoutSlider.value, 10);
                setStoredFadeoutTime(value);
                fadeoutValue.textContent = value === 0 ? 'never' : value + 's';
            };

            fadeoutRow.appendChild(fadeoutSlider);
            fadeoutRow.appendChild(fadeoutValue);

            // Draggable behavior
            makeDraggable(tools, 'uvnote-tools-pos', title);

            return tools;
        }

        function initTools() {
            const widget = createToolsWidget();
            document.body.appendChild(widget);
        }

        function teardownTools() {
            const w = document.querySelector('.tools-widget');
            if (w && w.parentNode) w.parentNode.removeChild(w);
        }

        // --- Canvas overlay for tools ---
        let _overlay = null;
        let _overlayCtx = null;
        let _overlayContainer = null; // window
        let _overlayMode = 'single';
        let _overlayResizeHandler = null;
        let _overlayScrollHandler = null;
        let _drawing = null; // current in-progress arrow {x1,y1,x2,y2}
        let _shapes = []; // committed shapes for current mode
        let _fadeTimer = null; // timer for fade animation

        function getOverlayStorageKey() { return 'uvnote-shapes'; }

        function loadShapes() {
            try {
                const raw = localStorage.getItem(getOverlayStorageKey());
                _shapes = raw ? JSON.parse(raw) : [];
            } catch (_) { _shapes = []; }
        }

        function saveShapes() {
            try { localStorage.setItem(getOverlayStorageKey(), JSON.stringify(_shapes)); } catch (_) {}
        }

        function updateShapesFade() {
            const now = Date.now();
            const fadeoutSeconds = getFadeoutTime();
            
            // If fadeout is disabled (0 seconds), don't fade anything
            if (fadeoutSeconds === 0) return;
            
            const fadeStartTime = Math.max(0, (fadeoutSeconds - 2) * 1000); // Start fading 2s before end
            const fadeEndTime = fadeoutSeconds * 1000; // Fully gone after specified time
            let needsUpdate = false;

            for (let i = _shapes.length - 1; i >= 0; i--) {
                const shape = _shapes[i];
                if (!shape.createdAt) continue; // Skip old shapes without timestamps
                
                const age = now - shape.createdAt;
                
                if (age >= fadeEndTime) {
                    // Remove completely faded shapes
                    _shapes.splice(i, 1);
                    needsUpdate = true;
                } else if (age >= fadeStartTime) {
                    // Update opacity for fading shapes
                    const fadeProgress = (age - fadeStartTime) / (fadeEndTime - fadeStartTime);
                    const newOpacity = 1 - fadeProgress;
                    if (Math.abs(shape.opacity - newOpacity) > 0.01) {
                        shape.opacity = newOpacity;
                        needsUpdate = true;
                    }
                }
            }

            if (needsUpdate) {
                saveShapes();
                renderOverlay();
            }
        }

        function getContentContainer() { return window; }

        function updateOverlayModeAndContainer() {
            _overlayContainer = window;
            _overlayMode = 'single';
        }

        function updateOverlayBounds() {
            if (!_overlay) return;
            if (_overlayContainer === window) {
                _overlay.style.position = 'fixed';
                _overlay.style.left = '0px';
                _overlay.style.top = '0px';
                _overlay.width = window.innerWidth;
                _overlay.height = window.innerHeight;
            } else {
                const rect = _overlayContainer.getBoundingClientRect();
                _overlay.style.position = 'fixed';
                _overlay.style.left = rect.left + 'px';
                _overlay.style.top = rect.top + 'px';
                _overlay.width = Math.max(0, Math.floor(rect.width));
                _overlay.height = Math.max(0, Math.floor(rect.height));
            }
            renderOverlay();
        }

        function containerScrollLeft() {
            return (_overlayContainer === window) ? (window.scrollX || 0) : (_overlayContainer.scrollLeft || 0);
        }
        function containerScrollTop() {
            return (_overlayContainer === window) ? (window.scrollY || 0) : (_overlayContainer.scrollTop || 0);
        }

        function toCanvasCoords(clientX, clientY) {
            const rect = _overlay.getBoundingClientRect();
            return { x: clientX - rect.left, y: clientY - rect.top };
        }

        function onPointerDown(e) {
            const tool = document.body.dataset.tool;
            if (tool === 'arrow') {
                startDrawArrow(e);
            } else if (tool === 'pen') {
                startDrawPen(e);
            } else if (tool === 'eraser') {
                eraseAt(e);
            } else if (tool === 'spotlight') {
                startDrawSpotlight(e);
            }
        }

        function onPointerMove(e) {
            // Update cursor position
            const pt = toCanvasCoords(e.touches ? e.touches[0].clientX : e.clientX, e.touches ? e.touches[0].clientY : e.clientY);
            _cursorX = pt.x;
            _cursorY = pt.y;
            
            if (!_drawing) {
                // Just update cursor position and re-render
                if (_cursorVisible) {
                    renderOverlay();
                }
                return;
            }
            
            if (_drawing.type === 'pen') {
                moveDrawPen(e);
            } else if (_drawing.type === 'spotlight') {
                moveDrawSpotlight(e);
            } else {
                moveDrawArrow(e);
            }
        }
        
        function onPointerEnter(e) {
            _cursorVisible = document.body.dataset.tool !== 'none';
            if (_cursorVisible) {
                renderOverlay();
            }
        }
        
        function onPointerLeave(e) {
            _cursorVisible = false;
            renderOverlay();
        }

        function onPointerUp(e) {
            if (!_drawing) return;
            if (_drawing.type === 'pen') {
                endDrawPen();
            } else if (_drawing.type === 'spotlight') {
                endDrawSpotlight();
            } else {
                endDrawArrow();
            }
        }

        function startDrawArrow(e) {
            if (document.body.dataset.tool !== 'arrow') return;
            const pt = toCanvasCoords(e.touches ? e.touches[0].clientX : e.clientX, e.touches ? e.touches[0].clientY : e.clientY);
            _drawing = {
                x1: pt.x + containerScrollLeft(),
                y1: pt.y + containerScrollTop(),
                x2: pt.x + containerScrollLeft(),
                y2: pt.y + containerScrollTop(),
                color: getArrowColor(),
                width: getLineThickness()
            };
            renderOverlay();
            e.preventDefault();
        }

        function moveDrawArrow(e) {
            if (!_drawing) return;
            const pt = toCanvasCoords(e.touches ? e.touches[0].clientX : e.clientX, e.touches ? e.touches[0].clientY : e.clientY);
            _drawing.x2 = pt.x + containerScrollLeft();
            _drawing.y2 = pt.y + containerScrollTop();
            renderOverlay();
            e.preventDefault();
        }

        function endDrawArrow() {
            if (!_drawing) return;
            _shapes.push({ 
                type: 'arrow', 
                ..._drawing,
                createdAt: Date.now(),
                opacity: 1.0
            });
            _drawing = null;
            saveShapes();
            renderOverlay();
        }

        function startDrawPen(e) {
            if (document.body.dataset.tool !== 'pen') return;
            const pt = toCanvasCoords(e.touches ? e.touches[0].clientX : e.clientX, e.touches ? e.touches[0].clientY : e.clientY);
            _drawing = {
                type: 'pen',
                points: [{
                    x: pt.x + containerScrollLeft(),
                    y: pt.y + containerScrollTop()
                }],
                color: getArrowColor(),
                width: getLineThickness()
            };
            renderOverlay();
            e.preventDefault();
        }

        function moveDrawPen(e) {
            if (!_drawing || _drawing.type !== 'pen') return;
            const pt = toCanvasCoords(e.touches ? e.touches[0].clientX : e.clientX, e.touches ? e.touches[0].clientY : e.clientY);
            _drawing.points.push({
                x: pt.x + containerScrollLeft(),
                y: pt.y + containerScrollTop()
            });
            renderOverlay();
            e.preventDefault();
        }

        function endDrawPen() {
            if (!_drawing || _drawing.type !== 'pen') return;
            if (_drawing.points.length > 1) {
                _shapes.push({ 
                    ..._drawing,
                    createdAt: Date.now(),
                    opacity: 1.0
                });
            }
            _drawing = null;
            saveShapes();
            renderOverlay();
        }

        function startDrawSpotlight(e) {
            if (document.body.dataset.tool !== 'spotlight') return;
            const pt = toCanvasCoords(e.touches ? e.touches[0].clientX : e.clientX, e.touches ? e.touches[0].clientY : e.clientY);
            _drawing = {
                type: 'spotlight',
                x: pt.x + containerScrollLeft(),
                y: pt.y + containerScrollTop(),
                radius: getLineThickness() * 20, // Use thickness to control spotlight size (bigger default)
                color: getArrowColor()
            };
            renderOverlay();
            e.preventDefault();
        }

        function moveDrawSpotlight(e) {
            if (!_drawing || _drawing.type !== 'spotlight') return;
            const pt = toCanvasCoords(e.touches ? e.touches[0].clientX : e.clientX, e.touches ? e.touches[0].clientY : e.clientY);
            const dx = pt.x + containerScrollLeft() - _drawing.x;
            const dy = pt.y + containerScrollTop() - _drawing.y;
            _drawing.radius = Math.max(20, Math.sqrt(dx * dx + dy * dy)); // Minimum radius of 20
            renderOverlay();
            e.preventDefault();
        }

        function endDrawSpotlight() {
            if (!_drawing || _drawing.type !== 'spotlight') return;
            _shapes.push({ 
                ..._drawing,
                createdAt: Date.now(),
                opacity: 1.0
            });
            _drawing = null;
            saveShapes();
            renderOverlay();
        }

        function distPointToSegment(px, py, x1, y1, x2, y2) {
            const dx = x2 - x1, dy = y2 - y1;
            if (dx === 0 && dy === 0) return Math.hypot(px - x1, py - y1);
            const t = Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy)));
            const cx = x1 + t * dx, cy = y1 + t * dy;
            return Math.hypot(px - cx, py - cy);
        }

        function eraseAt(e) {
            const pt = toCanvasCoords(e.touches ? e.touches[0].clientX : e.clientX, e.touches ? e.touches[0].clientY : e.clientY);
            const x = pt.x + containerScrollLeft();
            const y = pt.y + containerScrollTop();
            const threshold = 10; // pixels
            for (let i = _shapes.length - 1; i >= 0; i--) {
                const s = _shapes[i];
                if (s.type === 'arrow') {
                    const d = distPointToSegment(x, y, s.x1, s.y1, s.x2, s.y2);
                    if (d <= threshold) {
                        _shapes.splice(i, 1);
                        saveShapes();
                        renderOverlay();
                        break;
                    }
                } else if (s.type === 'pen' && s.points) {
                    // Check if click is near any line segment in the pen stroke
                    let minDist = Infinity;
                    for (let j = 1; j < s.points.length; j++) {
                        const d = distPointToSegment(x, y, s.points[j-1].x, s.points[j-1].y, s.points[j].x, s.points[j].y);
                        minDist = Math.min(minDist, d);
                    }
                    if (minDist <= threshold) {
                        _shapes.splice(i, 1);
                        saveShapes();
                        renderOverlay();
                        break;
                    }
                }
            }
            e.preventDefault();
        }

        function drawArrow(ctx, x1, y1, x2, y2, color, width, opacity = 1.0) {
            // Set opacity
            const oldAlpha = ctx.globalAlpha;
            ctx.globalAlpha = opacity;
            
            ctx.strokeStyle = color;
            ctx.fillStyle = color;
            ctx.lineWidth = width;
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
            
            // Check if points are too close (initial state)
            const dx = x2 - x1;
            const dy = y2 - y1;
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            if (distance < 5) {
                // Draw just a small arrowhead pointing down-right when first clicked
                const defaultAngle = Math.PI / 4; // 45 degrees (down-right)
                const headLength = Math.min(15 + width * 1.5, 25);
                const headAngle = Math.PI / 6;
                
                // Calculate arrowhead points
                const hx1 = x1 + headLength * Math.cos(defaultAngle - headAngle);
                const hy1 = y1 + headLength * Math.sin(defaultAngle - headAngle);
                const hx2 = x1 + headLength * Math.cos(defaultAngle + headAngle);
                const hy2 = y1 + headLength * Math.sin(defaultAngle + headAngle);
                
                // Draw arrowhead only
                ctx.beginPath();
                ctx.moveTo(x1, y1);
                ctx.lineTo(hx1, hy1);
                ctx.lineTo(hx2, hy2);
                ctx.closePath();
                ctx.fill();
            } else {
                // Normal arrow drawing - head at x1,y1, tail at x2,y2
                const angle = Math.atan2(y1 - y2, x1 - x2);
                const headLength = Math.min(15 + width * 1.5, 25);
                const headAngle = Math.PI / 6;
                
                // Calculate where the line should end (before the arrowhead)
                const lineEndX = x1 - headLength * 0.8 * Math.cos(angle);
                const lineEndY = y1 - headLength * 0.8 * Math.sin(angle);
                
                // Draw the line from tail to near the head
                ctx.beginPath();
                ctx.moveTo(x2, y2);
                ctx.lineTo(lineEndX, lineEndY);
                ctx.stroke();
                
                // Calculate arrowhead points
                const hx1 = x1 - headLength * Math.cos(angle - headAngle);
                const hy1 = y1 - headLength * Math.sin(angle - headAngle);
                const hx2 = x1 - headLength * Math.cos(angle + headAngle);
                const hy2 = y1 - headLength * Math.sin(angle + headAngle);
                
                // Draw arrowhead
                ctx.beginPath();
                ctx.moveTo(x1, y1);
                ctx.lineTo(hx1, hy1);
                ctx.lineTo(hx2, hy2);
                ctx.closePath();
                ctx.fill();
            }
            
            // Restore opacity
            ctx.globalAlpha = oldAlpha;
        }

        function drawPen(ctx, points, color, width, offX, offY, opacity = 1.0) {
            if (!points || points.length < 2) return;
            
            // Set opacity
            const oldAlpha = ctx.globalAlpha;
            ctx.globalAlpha = opacity;
            
            ctx.strokeStyle = color;
            ctx.lineWidth = width;
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
            ctx.beginPath();
            ctx.moveTo(points[0].x - offX, points[0].y - offY);
            for (let i = 1; i < points.length; i++) {
                ctx.lineTo(points[i].x - offX, points[i].y - offY);
            }
            ctx.stroke();
            
            // Restore opacity
            ctx.globalAlpha = oldAlpha;
        }

        function drawAllSpotlights(ctx, spotlights, offX, offY) {
            if (!spotlights || spotlights.length === 0) return;
            
            ctx.save();
            
            // Calculate the overall opacity based on all spotlights
            const maxOpacity = Math.max(...spotlights.map(s => s.opacity || 1.0));
            
            // Fill entire canvas with dark overlay
            ctx.fillStyle = `rgba(0, 0, 0, ${0.7 * maxOpacity})`;
            ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
            
            // Cut out completely transparent holes for all spotlights
            ctx.globalCompositeOperation = 'destination-out';
            ctx.fillStyle = 'rgba(0, 0, 0, 1)'; // Solid black to ensure complete removal
            for (const spotlight of spotlights) {
                ctx.beginPath();
                ctx.arc(spotlight.x - offX, spotlight.y - offY, spotlight.radius, 0, 2 * Math.PI);
                ctx.fill();
            }
            
            ctx.restore();
        }

        function renderOverlay() {
            if (!_overlay || !_overlayCtx) return;
            _overlayCtx.clearRect(0, 0, _overlay.width, _overlay.height);
            const offX = containerScrollLeft();
            const offY = containerScrollTop();
            // Draw non-spotlight shapes first
            for (const s of _shapes) {
                const opacity = s.opacity !== undefined ? s.opacity : 1.0;
                if (s.type === 'arrow') {
                    drawArrow(_overlayCtx, s.x1 - offX, s.y1 - offY, s.x2 - offX, s.y2 - offY, s.color || '#f00', s.width || 2, opacity);
                } else if (s.type === 'pen') {
                    drawPen(_overlayCtx, s.points, s.color || '#f00', s.width || 2, offX, offY, opacity);
                }
            }
            // Draw current drawing (non-spotlight)
            if (_drawing) {
                if (_drawing.type === 'pen') {
                    drawPen(_overlayCtx, _drawing.points, _drawing.color, _drawing.width, offX, offY);
                } else if (_drawing.type !== 'spotlight') {
                    drawArrow(_overlayCtx, _drawing.x1 - offX, _drawing.y1 - offY, _drawing.x2 - offX, _drawing.y2 - offY, _drawing.color, _drawing.width);
                }
            }
            
            // Collect all spotlights (existing + current drawing + cursor preview)
            const spotlights = [];
            
            // Add existing spotlight shapes
            for (const s of _shapes) {
                if (s.type === 'spotlight') {
                    spotlights.push({
                        x: s.x,
                        y: s.y, 
                        radius: s.radius,
                        opacity: s.opacity !== undefined ? s.opacity : 1.0
                    });
                }
            }
            
            // Add current spotlight being drawn
            if (_drawing && _drawing.type === 'spotlight') {
                spotlights.push({
                    x: _drawing.x,
                    y: _drawing.y,
                    radius: _drawing.radius,
                    opacity: 1.0
                });
            }
            
            // Add cursor preview spotlight if tool is active
            if (_cursorVisible && !_drawing) {
                const tool = document.body.dataset.tool;
                if (tool === 'spotlight') {
                    const thickness = getLineThickness();
                    const radius = thickness * 20;
                    const cursorWorldX = _cursorX + containerScrollLeft();
                    const cursorWorldY = _cursorY + containerScrollTop();
                    spotlights.push({
                        x: cursorWorldX,
                        y: cursorWorldY,
                        radius: radius,
                        opacity: 0.8
                    });
                }
            }
            
            // Draw all spotlights as a single overlay with multiple holes
            drawAllSpotlights(_overlayCtx, spotlights, offX, offY);
            
            // Draw cursor indicators for non-spotlight tools
            if (_cursorVisible && !_drawing) {
                const tool = document.body.dataset.tool;
                const color = getArrowColor();
                const thickness = getLineThickness();
                
                if (tool !== 'spotlight') {
                    _overlayCtx.save();
                    _overlayCtx.fillStyle = color;
                    _overlayCtx.globalAlpha = 0.7;
                    
                    if (tool === 'eraser') {
                        // Draw eraser indicator
                        _overlayCtx.strokeStyle = color;
                        _overlayCtx.lineWidth = 2;
                        _overlayCtx.beginPath();
                        _overlayCtx.arc(_cursorX, _cursorY, 10, 0, 2 * Math.PI);
                        _overlayCtx.stroke();
                    } else {
                        // Draw dot for pen/arrow
                        _overlayCtx.beginPath();
                        _overlayCtx.arc(_cursorX, _cursorY, thickness / 2, 0, 2 * Math.PI);
                        _overlayCtx.fill();
                    }
                    
                    _overlayCtx.restore();
                }
            }
        }

        function setOverlayActive(active) {
            if (!_overlay) initOverlay();
            _overlay.style.pointerEvents = active ? 'auto' : 'none';
            _overlay.style.cursor = active ? 'none' : 'auto';
            // Re-render to ensure visibility aligns with content
            renderOverlay();
        }

        function initOverlay() {
            if (_overlay) return;
            updateOverlayModeAndContainer();
            _overlay = document.createElement('canvas');
            _overlay.className = 'draw-overlay';
            _overlayCtx = _overlay.getContext('2d');
            document.body.appendChild(_overlay);
            updateOverlayBounds();
            loadShapes();
            renderOverlay();

            // Events
            _overlay.addEventListener('mousedown', onPointerDown);
            _overlay.addEventListener('mousemove', onPointerMove);
            _overlay.addEventListener('mouseenter', onPointerEnter);
            _overlay.addEventListener('mouseleave', onPointerLeave);
            document.addEventListener('mouseup', onPointerUp);
            _overlay.addEventListener('touchstart', onPointerDown, { passive: false });
            _overlay.addEventListener('touchmove', onPointerMove, { passive: false });
            document.addEventListener('touchend', onPointerUp);

            _overlayResizeHandler = () => updateOverlayBounds();
            window.addEventListener('resize', _overlayResizeHandler);

            _overlayScrollHandler = () => renderOverlay();
            window.addEventListener('scroll', _overlayScrollHandler);
            
            // Start fade animation timer
            _fadeTimer = setInterval(updateShapesFade, 100); // Update every 100ms for smooth fade
        }

        function rebindOverlayContainer() {
            if (!_overlay) return;
            // Remove old scroll handler
            if (_overlayScrollHandler) { window.removeEventListener('scroll', _overlayScrollHandler); }
            updateOverlayModeAndContainer();
            updateOverlayBounds();
            loadShapes();
            renderOverlay();
            _overlayScrollHandler = () => renderOverlay();
            window.addEventListener('scroll', _overlayScrollHandler);
        }

        function teardownOverlay() {
            if (!_overlay) return;
            _overlay.removeEventListener('mousedown', onPointerDown);
            _overlay.removeEventListener('mousemove', onPointerMove);
            _overlay.removeEventListener('mouseenter', onPointerEnter);
            _overlay.removeEventListener('mouseleave', onPointerLeave);
            document.removeEventListener('mouseup', onPointerUp);
            _overlay.removeEventListener('touchstart', onPointerDown);
            _overlay.removeEventListener('touchmove', onPointerMove);
            document.removeEventListener('touchend', onPointerUp);
            if (_overlayResizeHandler) window.removeEventListener('resize', _overlayResizeHandler);
            if (_overlayScrollHandler) {
                if (_overlayContainer === window) {
                    window.removeEventListener('scroll', _overlayScrollHandler);
                } else if (_overlayContainer) {
                    _overlayContainer.removeEventListener('scroll', _overlayScrollHandler);
                }
            }
            if (_fadeTimer) {
                clearInterval(_fadeTimer);
                _fadeTimer = null;
            }
            if (_overlay.parentNode) _overlay.parentNode.removeChild(_overlay);
            _overlay = null; _overlayCtx = null; _overlayContainer = null; _overlayResizeHandler = null; _overlayScrollHandler = null; _drawing = null;
        }
        
        function teardownFileExplorer() {
            const fe = document.querySelector('.file-explorer');
            if (fe && fe.parentNode) fe.parentNode.removeChild(fe);
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function runCell(cellId){
            const btn=document.querySelector('.run-btn[onclick*="'+cellId+'"]');
            const output=document.getElementById('output-'+cellId);
            if(btn){btn.textContent='⏳ running...';btn.disabled=true;}
            if(output){output.classList.add('output-stale');}
            fetch('/run/'+cellId,{method:'POST'}).then(r=>r.json()).then(data=>{
                if(output){
                    output.classList.remove('output-stale');
                    let html='';
                    if(data.stdout) html+='<div class="cell-stdout">'+escapeHtml(data.stdout)+'</div>';
                    console.log('UV Logs:', data);
                    if(data.stderr) {
                        // Split UV logs from regular stderr
                        const lines = data.stderr.split('\\n');
                        let uvLogs = [];
                        let regularLogs = [];
                        let inUvSection = true;

                        for (const line of lines) {
                            if (inUvSection) {
                                uvLogs.push(line);
                                if (line.startsWith('Installed ')) {
                                    inUvSection = false;
                                }
                            } else {
                                regularLogs.push(line);
                            }
                        }
                        

                        // If we never found "Installed", treat it all as regular stderr
                        if (inUvSection) {
                            html+='<div class="cell-stderr">'+escapeHtml(data.stderr)+'</div>';
                        } else {
                            const uvLogsStr = uvLogs.join('\\n');
                            const regularLogsStr = regularLogs.join('\\n').trim();

                            if (uvLogsStr) {
                                html+='<div class="uv-install-logs">';
                                html+='<div class="uv-logs-header" onclick="toggleUvLogs(this)">▶ UV Install Logs</div>';
                                html+='<div class="uv-logs-content" style="display: none;">'+escapeHtml(uvLogsStr)+'</div>';
                                html+='</div>';
                            }
                            if (regularLogsStr) {
                                html+='<div class="cell-stderr">'+escapeHtml(regularLogsStr)+'</div>';
                            }
                        }
                    }
                    output.innerHTML=html;
                }
                if(btn){btn.textContent='▶ run';btn.disabled=false;}
            }).catch(e=>{
                console.error('Run failed:',e);
                if(output){output.classList.remove('output-stale');}
                if(btn){btn.textContent='▶ run';btn.disabled=false;}
            });
        }

        function copyCell(cellId){
            console.log('copyCell called with cellId:', cellId);
            
            // Try multiple selectors to find the code element
            let codeElement = document.querySelector('#code-'+cellId+' code');
            if (!codeElement) {
                codeElement = document.querySelector('#code-'+cellId+' pre code');
            }
            if (!codeElement) {
                codeElement = document.querySelector('#code-'+cellId+' .highlight code');
            }
            if (!codeElement) {
                // Try finding any code element within the cell
                const codeDiv = document.getElementById('code-'+cellId);
                if (codeDiv) {
                    codeElement = codeDiv.querySelector('code');
                }
            }
            
            const btn = document.querySelector('.copy-btn[onclick*="'+cellId+'"]');
            
            console.log('Found codeElement:', codeElement);
            console.log('Found btn:', btn);
            console.log('Code div structure:', document.getElementById('code-'+cellId));
            
            if (!codeElement) {
                console.error('Code element not found for cell:', cellId);
                // Log the actual structure for debugging
                const codeDiv = document.getElementById('code-'+cellId);
                if (codeDiv) {
                    console.log('Code div HTML:', codeDiv.innerHTML);
                }
                return;
            }
            if (!btn) {
                console.error('Copy button not found for cell:', cellId);
                return;
            }
            
            const codeText = codeElement.textContent;
            console.log('Code text to copy:', codeText ? codeText.substring(0, 50) + '...' : 'empty');
            
            if (navigator.clipboard && navigator.clipboard.writeText) {
                navigator.clipboard.writeText(codeText).then(function() {
                    console.log('Clipboard copy successful');
                    btn.textContent = '✓ Copied!';
                    btn.classList.add('copied');
                    setTimeout(function() {
                        btn.textContent = 'Copy';
                        btn.classList.remove('copied');
                    }, 2000);
                }).catch(function(err) {
                    console.warn('Clipboard copy failed:', err);
                    fallbackCopy();
                });
            } else {
                console.log('Using fallback copy method');
                fallbackCopy();
            }
            
            function fallbackCopy() {
                const textarea = document.createElement('textarea');
                textarea.value = codeText;
                textarea.style.position = 'absolute';
                textarea.style.left = '-9999px';
                document.body.appendChild(textarea);
                textarea.select();
                try {
                    const success = document.execCommand('copy');
                    console.log('Fallback copy success:', success);
                    btn.textContent = '✓ Copied!';
                    btn.classList.add('copied');
                    setTimeout(function() {
                        btn.textContent = 'Copy';
                        btn.classList.remove('copied');
                    }, 2000);
                } catch (err) {
                    console.error('Fallback copy failed:', err);
                    btn.textContent = 'Copy failed';
                    setTimeout(function() {
                        btn.textContent = 'Copy';
                    }, 2000);
                }
                document.body.removeChild(textarea);
            }
        }

        // Live reload functionality (robust SSE handling)
        (function(){
            if (!('EventSource' in window)) {
                console.warn('SSE not supported in this browser');
                return;
            }
            let source = new EventSource('/events');
            let isOpen = false;
            source.onopen = function(){ isOpen = true; console.log('SSE connected'); };
            source.onmessage = function(e){
                const msg=(e.data||'').trim(); if(!msg) return;
                console.log('SSE message:', msg);
                if (msg==='reload' || msg==='incremental') { location.reload(); }
                // Ignore 'loading' to avoid premature reload loops
            };
            source.onerror = function(e){
                // Let EventSource auto-reconnect instead of forcing a reload
                if (isOpen) console.warn('SSE error after open, retrying...', e);
            };
            window.addEventListener('beforeunload', function(){ try{source.close();}catch(_){} });
        })();


        document.addEventListener('DOMContentLoaded', function() {
            updateThemeIcon();
            initMinimap();
            initFileExplorer();
            initTools();
            initOverlay();
            initializeWidgetVisibility();
            layoutWidgetsStackedBottomRight();
            window.addEventListener('resize', layoutWidgetsStackedBottomRight);

            // Add ESC key handler to exit tools
            document.addEventListener('keydown', function(e) {
                if (e.key === 'Escape' || e.keyCode === 27) {
                    const currentTool = document.body.dataset.tool;
                    if (currentTool && currentTool !== 'none') {
                        // Deactivate the current tool
                        window.setActiveTool('none');
                    }
                }
            });
        });
    </script>
</head>
<body>
    <div class="controls">
        <div class="theme-toggle" onclick="toggleTheme()">light</div>
        <div class="reset-toggle" onclick="resetLayout()">reset</div>
        <div class="menu-button" onclick="toggleMenu()">
            menu ▼
            <div class="menu-dropdown">
                <div class="menu-item" onclick="toggleWidget('tools')">
                    <span class="menu-checkbox" id="checkbox-tools">☐</span> Tools
                </div>
                <div class="menu-item" onclick="toggleWidget('file-explorer')">
                    <span class="menu-checkbox" id="checkbox-file-explorer">☐</span> File Explorer
                </div>
                <div class="menu-item" onclick="toggleWidget('minimap')">
                    <span class="menu-checkbox" id="checkbox-minimap">☐</span> Table of Contents
                </div>
            </div>
        </div>
    </div>
    
    <div class="system-info">
        <div class="system-info-header">Generated on:</div>
        <div class="system-info-content">
            {{ system_info.system }} {{ system_info.machine }} | {{ system_info.platform }}
        </div>
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
        linenos=config.show_line_numbers,
        linenos_special=1,
        cssclass="highlight",
    )
    return highlight(code, lexer, formatter)


def split_uv_install_logs(stderr: str) -> tuple[str, str]:
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


def render_cell(
    cell: CodeCell, result: ExecutionResult, highlighted_code: str, work_dir: Path
) -> str:
    """Render a single cell as HTML."""
    cell_class = "cell"
    if not result.success:
        cell_class += " cell-failed"
    if cell.commented:
        cell_class += " cell-commented"

    html_parts = [f'<div class="{cell_class}">']

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
    html_parts.append(f'<div id="code-{cell.id}" class="{code_class}">')
    html_parts.append(highlighted_code)
    html_parts.append("</div>")

    # Cell output - handle collapse state
    output_class = "cell-output"
    if cell.collapse_output:
        output_class += " collapsed"
    html_parts.append(f'<div id="output-{cell.id}" class="{output_class}">')

    if result.stdout:
        if getattr(result, "is_html", False):
            html_parts.append(f'<div class="cell-stdout">{result.stdout}</div>')
        else:
            html_parts.append(
                f'<div class="cell-stdout">{html.escape(result.stdout)}</div>'
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

        # Image previews
        cache_dir = work_dir / ".uvnote" / "cache"
        for artifact in result.artifacts:
            if artifact.endswith((".png", ".jpg", ".jpeg")):
                html_parts.append('<div class="artifact-preview">')
                html_parts.append(
                    f'<img src="artifacts/{cell.id}/{artifact}" alt="{artifact}">'
                )
                html_parts.append("</div>")
            elif artifact.endswith(".svg"):
                # Read and embed SVG content directly
                svg_path = cache_dir / result.cache_key / artifact
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
                                f'<img src="artifacts/{cell.id}/{artifact}" alt="{artifact}">'
                            )
                            html_parts.append("</div>")
                    except Exception:
                        # Fallback to img tag on error
                        html_parts.append('<div class="artifact-preview">')
                        html_parts.append(
                            f'<img src="artifacts/{cell.id}/{artifact}" alt="{artifact}">'
                        )
                        html_parts.append("</div>")

        html_parts.append("</div>")

    html_parts.append("</div>")
    html_parts.append("</div>")

    return "\n".join(html_parts)


def generate_html(
    markdown_content: str,
    config: DocumentConfig,
    cells: List[CodeCell],
    results: List[ExecutionResult],
    output_path: Path,
    work_dir: Path,
) -> None:
    """Generate static HTML from markdown content and execution results."""

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
                            cell, result, highlighted_code, work_dir
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
    content_html = md.convert(clean_content)

    # Setup Jinja2 environment
    env = Environment(loader=BaseLoader())
    template = env.from_string(HTML_TEMPLATE)

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
