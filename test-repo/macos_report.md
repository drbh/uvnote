---
title: macOS System Report
platforms:
  - darwin
---

# macOS System Report

This report only runs on macOS (darwin) systems.

## System Information

```python id=sysinfo
import platform
import subprocess

print(f"System: {platform.system()}")
print(f"Release: {platform.release()}")
print(f"Version: {platform.version()}")
print(f"Machine: {platform.machine()}")
print(f"Processor: {platform.processor()}")
```

## macOS-Specific Info

```python id=macos_info needs=sysinfo
import subprocess

# Get macOS version name
result = subprocess.run(["sw_vers"], capture_output=True, text=True)
print("Software Version:")
print(result.stdout)

# Get hardware overview
result = subprocess.run(["system_profiler", "SPHardwareDataType"], capture_output=True, text=True)
print("\nHardware Overview:")
for line in result.stdout.split('\n')[:15]:
    if line.strip():
        print(line)
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
