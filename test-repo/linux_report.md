---
title: Linux System Report
platforms:
  - linux
---

# Linux System Report

This report only runs on Linux systems.

## System Information

```python id=sysinfo
import platform
import os

print(f"System: {platform.system()}")
print(f"Release: {platform.release()}")
print(f"Version: {platform.version()}")
print(f"Machine: {platform.machine()}")
print(f"Processor: {platform.processor()}")
```

## Linux-Specific Info

```python id=linux_info needs=sysinfo
import subprocess

# Get distribution info
try:
    with open("/etc/os-release") as f:
        print("Distribution Info:")
        for line in f:
            if line.startswith(("NAME=", "VERSION=", "ID=", "PRETTY_NAME=")):
                print(f"  {line.strip()}")
except FileNotFoundError:
    print("Could not read /etc/os-release")

# Get kernel info
result = subprocess.run(["uname", "-a"], capture_output=True, text=True)
print(f"\nKernel: {result.stdout.strip()}")
```

## Memory Info

```python id=memory
with open("/proc/meminfo") as f:
    lines = f.readlines()[:5]
    print("Memory Info:")
    for line in lines:
        print(f"  {line.strip()}")
```

## Disk Usage

```python id=disk
import shutil

total, used, free = shutil.disk_usage("/")
print(f"Total: {total // (1024**3)} GB")
print(f"Used:  {used // (1024**3)} GB")
print(f"Free:  {free // (1024**3)} GB")
print(f"Usage: {used / total * 100:.1f}%")
```
