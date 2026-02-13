# HDF5 reader using h5dump (correctly handles nested VLEN, jagged arrays, etc.)

import subprocess
import shutil
import sys

def show_h5(filename, h5dump=None):
    """Dump HDF5 file contents using h5dump."""
    if h5dump is None:
        h5dump = shutil.which('h5dump') or 'h5dump'
    result = subprocess.run(
        [h5dump, filename],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode)
    print(result.stdout)

if __name__ == '__main__':
    show_h5(sys.argv[1])
