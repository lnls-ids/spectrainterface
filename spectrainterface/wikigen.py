"""Graphs generator functions."""

import numpy as _np
import matplotlib.pyplot as _plt
from matplotlib import colors
import matplotlib.patches as _patches
from spectrainterface.interface import SpectraInterface
import copy
import mathphys
import importlib
import inspect

ECHARGE = mathphys.constants.elementary_charge
EMASS = mathphys.constants.electron_mass
LSPEED = mathphys.constants.light_speed
ECHARGE_MC = ECHARGE / (2 * _np.pi * EMASS * LSPEED)
PLANCK = mathphys.constants.reduced_planck_constant
VACUUM_PERMITTICITY = mathphys.constants.vacuum_permitticity
PI = _np.pi
