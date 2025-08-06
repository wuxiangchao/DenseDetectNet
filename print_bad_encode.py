import os

print("--- Starting scan for files with non-UTF-8 encoding ---")
failed_files = []

# 遍历当前目录及所有子目录
for root, dirs, files in os.walk("."):
    # 排除虚拟环境目录，避免扫描其中的库文件
    if ".venv" in dirs:
        dirs.remove(".venv")
    if "venv" in dirs:
        dirs.remove("venv")
    if "__pycache__" in dirs:
        dirs.remove("__pycache__")

    for file in files:
        # 只检查 .py 文件
        if file.endswith(".py"):
            filepath = os.path.join(root, file)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    f.read()
                print(f"OK: {filepath}")
            except UnicodeDecodeError:
                # 找到了！这就是我们要找的文件
                print(f"!!! FAILED TO READ: {filepath} <--- THIS IS THE PROBLEM FILE")
                failed_files.append(filepath)
            except Exception as e:
                print(f"An unexpected error occurred with {filepath}: {e}")

print("\n--- Scan Complete ---")
if failed_files:
    print("Found files with encoding issues:")
    for f in failed_files:
        print(f"- {f}")
else:
    print("No files with UTF-8 encoding issues were found.")
