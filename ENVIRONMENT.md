# Python environment

Use the single `.venv` environment for ROS control, GUIs, development, and
OpenMV uploads. It uses Python 3.8 with `--system-site-packages` to match
ROS Noetic and this machine's native `PyQt5.Qwt` bindings. It reuses the
installed scientific, ROS, and Qt packages, including user-site packages,
and adds `mpremote` locally. It is not isolated from changes to those shared
packages.

For ROS control and GUI work, run from the workspace root:

```bash
source /opt/ros/noetic/setup.bash
source devel/setup.bash
source .venv/bin/activate
python --version
mpremote --version
```

In VS Code, use **Python: Select Interpreter** and select
`.venv/bin/python`. Existing terminal sessions may
need to be reactivated. A script with a fixed `/usr/bin/python3` shebang
continues to use that interpreter; activation does not override it.

To recreate the environment on this machine:

```bash
/usr/bin/python3 -m venv --without-pip --system-site-packages .venv
.venv/bin/python -m pip install 'mpremote==1.25.0'
```

This recreation command relies on pip and other dependencies already installed
for the system Python. On another machine, install the matching system ROS/Qt
packages and Python venv/pip tools first. `requirements.txt` lists host-side
pip dependencies. ROS modules require the ROS installation and its setup script;
`PyQt5.Qwt` requires matching native bindings. The pinned polytope libraries
use the older cdd API required by `pypoman` 1.1.1; compiling `pycddlib` may
require a C compiler and GMP development headers.

The camera's `sensor`, `image`, and `pyb` modules run on OpenMV firmware and
must not be installed as desktop Python packages. Release the camera serial
port from the ROS tracker or IDE before using `mpremote`.

Validation covered dependency imports and headless construction of all nine
packages' generated control panels in `.venv`; no CAN commands or camera
uploads were performed. The shared Python 3.8 installation has unrelated
`pip check` warnings for Orbbec SDK extras, `launchpadlib`, and a Jupyter version
mismatch. Those shared packages were not changed as part of this setup.
