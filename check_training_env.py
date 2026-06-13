import subprocess
import sys
import platform

print("=== SYSTEM INFO ===")
print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")
print()

print("=== GPU INFO ===")
try:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
         "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10
    )
    if result.returncode == 0:
        print(result.stdout)
    else:
        print("nvidia-smi failed:", result.stderr)
except FileNotFoundError:
    print("nvidia-smi not found - no NVIDIA GPU or drivers not installed")
except Exception as e:
    print(f"Error: {e}")

print()
print("=== INSTALLED PACKAGES (relevant) ===")
relevant = ["torch", "transformers", "trl", "peft", "accelerate",
            "bitsandbytes", "datasets", "huggingface_hub"]
for pkg in relevant:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", pkg],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if line.startswith("Name:") or line.startswith("Version:"):
                    print(line)
        else:
            print(f"{pkg}: NOT INSTALLED")
    except Exception as e:
        print(f"{pkg}: error - {e}")

print()
print("=== RAM ===")
try:
    import ctypes
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(stat)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    print(f"Total RAM: {stat.ullTotalPhys / (1024**3):.1f} GB")
    print(f"Available RAM: {stat.ullAvailPhys / (1024**3):.1f} GB")
except Exception as e:
    print(f"Could not get RAM info: {e}")
