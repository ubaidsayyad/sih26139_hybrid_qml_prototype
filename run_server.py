"""
run_server.py (Root Launcher)
-----------------------------
Launches the Waitress WSGI production server from the project root.
"""
import os
import sys

root_dir = os.path.dirname(os.path.abspath(__file__))
sub_dir = os.path.join(root_dir, "sih_qml_prototype")
if sub_dir not in sys.path:
    sys.path.insert(0, sub_dir)

from sih_qml_prototype.run_server import main

if __name__ == "__main__":
    main()
