import os
import sys
import time
import subprocess

# Folders and files to exclude from monitoring
EXCLUDE_DIRS = {".git", "venv", "assets", "snapshots", "__pycache__"}
EXCLUDE_FILES = {"auto_sync.py"}

def get_files_mtimes(root_dir):
    """Recursively retrieves modification times of monitored files."""
    mtimes = {}
    for root, dirs, files in os.walk(root_dir):
        # Modify dirs in-place to skip excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for file in files:
            if file in EXCLUDE_FILES or file.endswith((".pyc", ".pyo", ".pyd")):
                continue
            filepath = os.path.join(root, file)
            try:
                mtimes[filepath] = os.path.getmtime(filepath)
            except Exception:
                pass
    return mtimes

def git_sync():
    """Performs git add, commit, and push operations."""
    print(f"[{time.strftime('%H:%M:%S')}] File changes detected! Synchronizing with GitHub...")
    try:
        # 1. Stage all changes
        subprocess.run(["git", "add", "."], check=True, capture_output=True)
        
        # 2. Commit changes
        commit_msg = f"Auto-update: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        subprocess.run(["git", "commit", "-m", commit_msg], capture_output=True)
        
        # 3. Push to main
        result = subprocess.run(["git", "push", "origin", "main"], check=True, capture_output=True, text=True)
        print(f"[{time.strftime('%H:%M:%S')}] Push successful! GitHub updated.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[{time.strftime('%H:%M:%S')}] ERROR: Git sync failed:")
        if e.stderr:
            print(e.stderr.strip())
        return False

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    print("==================================================")
    print("      GitHub Auto-Sync Service Initialized        ")
    print("==================================================")
    print("Watching for code modifications, additions, or deletions...")
    print("Press Ctrl+C to stop.")
    
    # Store initial state of modification times
    last_mtimes = get_files_mtimes(root_dir)
    
    try:
        while True:
            time.sleep(5)  # Scan every 5 seconds
            current_mtimes = get_files_mtimes(root_dir)
            
            # Check if any files were added, deleted, or modified
            has_changes = False
            if current_mtimes.keys() != last_mtimes.keys():
                has_changes = True
            else:
                for path, mtime in current_mtimes.items():
                    if last_mtimes.get(path) != mtime:
                        has_changes = True
                        break
            
            if has_changes:
                git_sync()
                last_mtimes = current_mtimes
    except KeyboardInterrupt:
        print("\nAuto-Sync Service stopped.")

if __name__ == "__main__":
    main()
