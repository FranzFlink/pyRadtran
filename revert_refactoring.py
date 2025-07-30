#!/usr/bin/env python3
"""
Revert script for pyradtran refactoring.

This script reverts all changes made by the migration script.
Run this if you need to go back to the original system.
"""

import shutil
from pathlib import Path

def revert_file(original_path, backup_suffix='.backup'):
    """Revert a single file from backup"""
    original = Path(original_path)
    backup = original.with_suffix(original.suffix + backup_suffix)
    
    if backup.exists():
        shutil.copy2(backup, original)
        print(f"✓ Reverted {original}")
        return True
    else:
        print(f"✗ Backup not found: {backup}")
        return False

def main():
    """Revert all migrated files"""
    print("=" * 50)
    print("REVERTING PYRADTRAN REFACTORING")
    print("=" * 50)
    
    files_to_revert = [
        "pyradtran/config.py",
        "pyradtran/io.py", 
        "pyradtran/core.py",
        "pyradtran/interface.py",
        "pyradtran/__init__.py"
    ]
    
    success_count = 0
    for file_path in files_to_revert:
        if revert_file(file_path):
            success_count += 1
    
    print(f"\n✓ Reverted {success_count}/{len(files_to_revert)} files")
    
    if success_count == len(files_to_revert):
        print("🔄 All files reverted successfully!")
        print("\nTo complete the revert:")
        print("1. git add .")
        print("2. git commit -m 'Revert refactoring changes'")  
        print("\nOr use: git reset --hard backup-before-refactoring")
    else:
        print("❌ Some files could not be reverted. Check the backup files manually.")

if __name__ == "__main__":
    main()
