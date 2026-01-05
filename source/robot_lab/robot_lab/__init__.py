# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
Python module serving as a project/extension template.
"""

# Register Gym environments.
# from .tasks import *
try:
    import omni.kit.app  # noqa: F401
except ModuleNotFoundError:
    # 允许在非 Isaac Sim 环境下 import robot_lab.utils
    pass
else:
    from .tasks import *  # noqa


# Register UI extensions.
# from .ui_extension_example import *

try:
    import omni.kit.app  # noqa: F401
except ModuleNotFoundError:
    # 允许在非 Isaac Sim 环境下 import robot_lab.utils
    pass
else:
    from .ui_extension_example import *
