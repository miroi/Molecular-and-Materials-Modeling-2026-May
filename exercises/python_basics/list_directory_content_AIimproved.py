#
# MS Copolitor improved code
#
from pathlib import Path

def format_size(size_in_bytes):
    """Convert bytes into human-readable units (KB, MB, GB, TB)."""
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    size = float(size_in_bytes)
    for unit in units:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"  # fallback for very large files

def list_files_by_size(directory_path, descending=True):
    path = Path(directory_path)
    if not path.is_dir():
        raise ValueError(f"{directory_path} is not a valid directory.")

    # Collect file names and sizes once (avoid repeated stat calls)
    files_with_size = [(file.name, file.stat().st_size) for file in path.iterdir() if file.is_file()]
    files_with_size.sort(key=lambda x: x[1], reverse=descending)

    # Print header
    print(f"{'File Name':<40} | {'Size':<12}")
    print("-" * 55)
    for name, size in files_with_size:
        print(f"{name:<40} | {format_size(size):<12}")

    return files_with_size

# Example usage
list_files_by_size('.')

