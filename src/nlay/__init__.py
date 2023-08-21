"""
NPad Layouting Library
=====

NLay (pronounced "Enlay") is layouting library inspired by the flexibility of Android's
[ConstraintLayout](https://developer.android.com/training/constraint-layout). This layouting library attempts to
implement subset of the ConstraintLayout layouting functionality. NLay is only meant to help you place element on the
screen.

This Python implementation is 1:1 mapping within its [Lua](https://github.com/MikuAuahDark/NPad93#nlay) counterpart.

Importing this module is as simple as `import nlay`. Please don't use `from nlay import *`!
"""

from .nlay import (
    BaseConstraint,
    UnitMode,
    LineDirection,
    Constraint,
    Inside,
    MaxConstraint,
    LineConstraint,
    GridCellConstraint,
    Grid,
    get,
    update,
    Inside as inside,
    MaxConstraint as max,
    LineConstraint as line,
    Grid as grid,
)
