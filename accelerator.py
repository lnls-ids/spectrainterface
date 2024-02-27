"""Storage ring parameters."""
import numpy as _np
import matplotlib.pyplot as _plt


class StorageRingParameters:
    """Class with storage ring parameters for radiation calculations."""

    def __init__(self):
        """Class constructor."""
        self._energy = 3  # [GeV]
        self._current = 100  # [mA]
        self._sigmaz = 2.9  # [mm]
        self._nat_emittance = 2.5e-10  # [m rad]
        self._coupling_constant = 0.01
        self._energy_spread = 0.00084
        self._betax = 1.499  # [m]
        self._betay = 1.435  # [m]
        self._alphax = 0
        self._alphay = 0
        self._etax = 0  # [m]
        self._etay = 0  # [m]
        self._etapx = 0
        self._etapy = 0

        self._zero_emittance = False
        self._zero_energy_spread = False
        self._injection_condition = 'Align at Entrance'
