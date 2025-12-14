from pathlib import Path
from pyradtran.config import load_config
import yaml
import os

# Create a dummy config with tilde path
config_content = {
    'paths': {
        'radiosonde_base': '~/pyRadtran/book/notebooks/radiosonde'
    }
}
with open('test_tilde_config.yaml', 'w') as f:
    yaml.dump(config_content, f)

try:
    config = load_config('test_tilde_config.yaml')
    path = config.paths.radiosonde_base
    print(f"Resolved path: {path}")
    print(f"Is absolute? {path.is_absolute()}")
    print(f"Expected: {Path('~/pyRadtran/book/notebooks/radiosonde').expanduser()}")
    if path == Path('~/pyRadtran/book/notebooks/radiosonde').expanduser():
        print("SUCCESS")
    else:
        print("FAILURE")
except Exception as e:
    print(f"Failed: {e}")
finally:
    if os.path.exists('test_tilde_config.yaml'):
        os.remove('test_tilde_config.yaml')
