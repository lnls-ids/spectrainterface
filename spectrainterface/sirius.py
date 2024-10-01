"""SIRIUS parameters."""

import numpy as _np
import matplotlib.pyplot as _plt
import mathphys.constants as _constants
from spectrainterface.accelerator import StorageRingParameters

ECHARGE = _constants.elementary_charge
EREST = _constants.electron_rest_energy


class SIRIUS:
    """Class with SIRIUS parameters for radiation calculations."""

    class StorageRing(StorageRingParameters):
        def __init__(self):
            """Class constructor."""
            self._energy = 3  # [GeV]
            self._current = 100  # [mA]
            self._sigmaz = 2.9  # [mm]
            self._nat_emittance = 2.5e-10  # [m rad]
            self._coupling_constant = 0.01
            self._energy_spread = 0.00084
            self._gamma = self._energy / (1e-9 * EREST / ECHARGE)
            self._betax = 1.499  # [m]
            self._betay = 1.435  # [m]
            self._alphax = 0
            self._alphay = 0
            self._etax = 0  # [m]
            self._etay = 0  # [m]
            self._etapx = 0
            self._etapy = 0
            self._beta_section = "low"

            self._zero_emittance = False
            self._zero_energy_spread = False
            self._injection_condition = "Align at Entrance"

            # BSC parameters
            self._bsc0_h_lowbeta = 3.4529
            self._bsc0_v_lowbeta = 1.5588
            self._bsc0_h_highbeta = 11.6952
            self._bsc0_v_highbeta = 2.4706
            self._bsc0_h = 3.4529
            self._bsc0_v = 1.5588

        @property
        def bsc0_h_highbeta(self):
            """Horizontal BSC at center of high beta section.

            Returns:
                float: Horizontal BSC High Beta [mm]
            """
            return self._bsc0_h_highbeta

        @property
        def bsc0_v_highbeta(self):
            """Vertical BSC at center of high beta section.

            Returns:
                float: Vertical BSC High Beta [mm]
            """
            return self._bsc0_v_highbeta

        @property
        def bsc0_h_lowbeta(self):
            """Horizontal BSC at center of low beta section.

            Returns:
                float: Horizontal BSC low Beta [mm]
            """
            return self._bsc0_h_lowbeta

        @property
        def bsc0_v_lowbeta(self):
            """Vertical BSC at center of low beta section.

            Returns:
                float: Vertical BSC low Beta [mm]
            """
            return self._bsc0_v_lowbeta

        @property
        def beta_section(self):
            """Beta section.

            Returns:
                str: Beta section (high, low or other)
            """
            return self._beta_section

        def set_current_bsc(self):
            """Set current BSC (03/07/2024)."""
            self._bsc0_h_lowbeta = 3.4529
            self._bsc0_v_lowbeta = 1.8627
            self._bsc0_h_highbeta = 11.6952
            self._bsc0_v_highbeta = 2.9524
            self._update_bsc()

        def set_bsc_orion_reduction(self):
            """Set BSC after changes due to ORION."""
            self._bsc0_h_lowbeta = 3.4529
            self._bsc0_v_lowbeta = 1.38
            self._bsc0_h_highbeta = 11.6952
            self._bsc0_v_highbeta = 2.18
            self._update_bsc()

        def set_low_beta_section(self):
            """Set low beta section."""
            self.betax = 1.499
            self.betay = 1.435
            self.alphax = 0
            self.alphay = 0
            self.etax = 0
            self.etay = 0
            self.etapx = 0
            self.etapy = 0
            self._beta_section = "low"
            self._bsc0_h = self._bsc0_h_lowbeta
            self._bsc0_v = self._bsc0_v_lowbeta

        def set_high_beta_section(self):
            """Set high beta section."""
            self.betax = 17.20
            self.betay = 3.605
            self.alphax = 0
            self.alphay = 0
            self.etax = 0
            self.etay = 0
            self.etapx = 0
            self.etapy = 0
            self._beta_section = "high"
            self._bsc0_h = self._bsc0_h_highbeta
            self._bsc0_v = self._bsc0_v_highbeta

        def set_bc_section(self):
            """Set bc section section."""
            self.betax = 0.338
            self.betay = 5.356
            self.alphax = 0.003
            self.alphay = 0
            self.etax = 0.002
            self.etay = 0
            self.etapx = 0
            self.etapy = 0
            self._beta_section = "bc"

        def set_b1_section(self):
            """Set b1 section section."""
            self.betax = 1.660
            self.betay = 26.820
            self.alphax = 2.908
            self.alphay = -6.564
            self.etax = 0.122e-3
            self.etay = 0
            self.etapx = 3.211e-3
            self.etapy = 0
            self._beta_section = "b1"

        def set_b2_section(self):  # It is necessary to update these values.
            """Set b2 section section."""
            self.betax = 1.265
            self.betay = 25.5
            self.alphax = 1.94
            self.alphay = 0
            self.etax = 0.025
            self.etay = 0
            self.etapx = 0
            self.etapy = 0
            self._beta_section = "b2"
        
        def _update_bsc(self):
            if self.beta_section in ['low', 'high']:
                if self.beta_section == 'low':
                    self._bsc0_h = self._bsc0_h_lowbeta
                    self._bsc0_v = self._bsc0_v_lowbeta
                elif self.beta_section == 'high':
                    self._bsc0_h = self._bsc0_h_highbeta
                    self._bsc0_v = self._bsc0_v_highbeta
            else:
                raise ValueError("A beta section must be selected")
