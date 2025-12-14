
import yaml
from pathlib import Path
import os

CONFIG_DIR = Path("/home/josh/pyRadtran/book/config")

def refactor_config(file_path):
    print(f"Processing {file_path}...")
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            # Preserve comments is hard with pure PyYAML. 
            # We want to be careful not to destroy comments.
            # But the requirement is to remove specific keys.
            # Let's try to interpret the structure using PyYAML first to see what to remove.
            # But parsing and dumping destroys structure/comments.
            # Given the simple structure, maybe we can just read the YAML, modify the dict, and dump it if we accept reformatting.
            # OR we can assume line-based regex removal if we want to be super surgical.
            # But "make minimal" implies reformatting is probably fine, assuming comments are less critical or we try to keep them.
            # Let's try parsing with ruamel.yaml if available, or just PyYAML safe_dumper.
            # Wait, I don't know if ruamel.yaml is installed.
            # Let's stick to safe string removal first if possible or accept PyYAML formatting.
            
            # Actually, standard pyyaml cleans up everything. The user said "adapt each config".
            # Most users prefer comments to stay. 
            # Let's try to load, modify, and dump. If comments are lost, that's a trade-off.
            # The user's goal is functionality.
            pass

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    # Loading with PyYAML
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f) or {}

    if 'paths' not in data:
        print(f"  No 'paths' section in {file_path.name}")
        return

    paths = data.get('paths', {})
    changed = False

    # 1. Remove binary and data dir
    if 'libradtran_bin' in paths:
        del paths['libradtran_bin']
        changed = True
    
    if 'libradtran_data' in paths:
        del paths['libradtran_data']
        changed = True

    # 2. Check atmosphere profile
    atm = paths.get('atmosphere_profile')
    if atm and 'afglus.dat' in str(atm):
        del paths['atmosphere_profile']
        changed = True
        print(f"  Removed default atmosphere_profile: {atm}")
    
    # 3. Check solar spectrum
    solar = paths.get('solar_spectrum')
    if solar and 'kurudz_1.0nm.dat' in str(solar):
        del paths['solar_spectrum']
        changed = True
        print(f"  Removed default solar_spectrum: {solar}")
    
    # Clean up empty paths
    if not paths:
        del data['paths']
        changed = True
    
    if changed:
        print(f"  Writing changes to {file_path.name}")
        with open(file_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    else:
        print(f"  No changes needed for {file_path.name}")

def main():
    if not CONFIG_DIR.exists():
        print(f"Directory not found: {CONFIG_DIR}")
        return

    for config_file in CONFIG_DIR.glob("*.yaml"):
        refactor_config(config_file)

if __name__ == "__main__":
    main()
