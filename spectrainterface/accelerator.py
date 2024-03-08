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
        self._injection_condition = "Align at Entrance"

        # BSC parameters
        self._bsc0_h = 0
        self._bsc0_v = 0

        self._bsc0_h_lowbeta = 3.4529
        self._bsc0_v_lowbeta = 1.8627
        self._bsc0_h_highbeta = 11.6952
        self._bsc0_v_highbeta = 2.9524

    @property
    def energy(self):
        """Accelerator energy.

        Returns:
            float: Energy [GeV]
        """
        return self._energy

    @property
    def current(self):
        """Storage ring current.

        Returns:
            float: Current [mA]
        """
        return self._current

    @property
    def sigmaz(self):
        """Bunch length.

        Returns:
            float: Bunch length [mm]
        """
        return self._sigmaz

    @property
    def nat_emittance(self):
        """Natural emittance.

        Returns:
            float: Natural emittance [m rad]
        """
        return self._nat_emittance

    @property
    def coupling_constant(self):
        """Coupling constant.

        Returns:
            float: Coupling constant
        """
        return self._coupling_constant

    @property
    def energy_spread(self):
        """Energy spread.

        Returns:
            float: Energy spread
        """
        return self._energy_spread

    @property
    def betax(self):
        """Horizontal beta function.

        Returns:
            float: betax [m]
        """
        return self._betax

    @property
    def betay(self):
        """Vertical beta function.

        Returns:
            float: betay [m]
        """
        return self._betay

    @property
    def alphax(self):
        """Horizontal alpha function.

        Returns:
            float: alphax [m]
        """
        return self._alphax

    @property
    def alphay(self):
        """Vertical alpha function.

        Returns:
            float: alphay [m]
        """
        return self._alphay

    @property
    def etax(self):
        """Horizontal dispersion function.

        Returns:
            float: etax [m]
        """
        return self._etax

    @property
    def etay(self):
        """Vertical dispersion function.

        Returns:
            float: etay [m]
        """
        return self._etay

    @property
    def etapx(self):
        """Derivative of horizontal dispersion function.

        Returns:
            float: etax [m]
        """
        return self._etapx

    @property
    def etapy(self):
        """Derivative of vertical dispersion function.

        Returns:
            float: etay [m]
        """
        return self._etapy

    @property
    def zero_emittance(self):
        """Use zero emittance.

        Returns:
            Boolean: Use beam with zero emittance
        """
        return self._zero_emittance

    @property
    def zero_energy_spread(self):
        """Use zero energy spread.

        Returns:
            Boolean: Use beam with zero energy spread
        """
        return self._zero_energy_spread

    @property
    def injection_condition(self):
        """Injection condition.

        Returns:
            string: Initial condition of electron in magnetic fields.
        """
        return self._injection_condition

    @property
    def bsc0_h(self):
        """Hozizontal Beam Stay Clear at center of straight section.

        Returns:
            float: Horizontal BSC [mm]
        """
        return self._bsc0_h

    @property
    def bsc0_v(self):
        """Vertical Beam Stay Clear at center of straight section.

        Returns:
            float: Vertical BSC [mm]
        """
        return self._bsc0_v

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

    @energy.setter
    def energy(self, value):
        self._energy = value

    @current.setter
    def current(self, value):
        self._current = value

    @sigmaz.setter
    def sigmaz(self, value):
        self._sigmaz = value

    @nat_emittance.setter
    def nat_emittance(self, value):
        self._nat_emittance = value

    @coupling_constant.setter
    def coupling_constant(self, value):
        self._coupling_constant = value

    @energy_spread.setter
    def energy_spread(self, value):
        self._energy_spread = value

    @betax.setter
    def betax(self, value):
        self._betax = value

    @betay.setter
    def betay(self, value):
        self._betay = value

    @alphax.setter
    def alphax(self, value):
        self._alphax = value

    @alphay.setter
    def alphay(self, value):
        self._alphay = value

    @etax.setter
    def etax(self, value):
        self._etax = value

    @etay.setter
    def etay(self, value):
        self._etay = value

    @etapx.setter
    def etapx(self, value):
        self._etapx = value

    @etapy.setter
    def etapy(self, value):
        self._etapy = value

    @zero_emittance.setter
    def zero_emittance(self, value):
        if type(value) is not bool:
            raise ValueError("Argument must be bool!")
        else:
            self._zero_emittance = value

    @zero_energy_spread.setter
    def zero_energy_spread(self, value):
        if type(value) is not bool:
            raise ValueError("Argument must be bool!")
        else:
            self._zero_energy_spread = value

    @injection_condition.setter
    def injection_condition(self, value):
        if type(value) is not str:
            raise ValueError("Argument must be str!")
        else:
            self._injection_condition = value

    @bsc0_h.setter
    def bsc0_h(self, value):
        self._bsc0_h = value

    @bsc0_v.setter
    def bsc0_v(self, value):
        self._bsc0_v = value

    def set_current_bsc(self):
        """Set current BSC (03/07/2024)."""
        self._bsc0_h_lowbeta = 3.4529
        self._bsc0_v_lowbeta = 1.8627
        self._bsc0_h_highbeta = 11.6952
        self._bsc0_v_highbeta = 2.9524

    def set_bsc_with_ivu18(self):
        """Set BSC with IVU18."""
        self._bsc0_h_lowbeta = 3.4529
        self._bsc0_v_lowbeta = 1.6818
        self._bsc0_h_highbeta = 11.6952
        self._bsc0_v_highbeta = 2.6658

    def set_low_beta_section(self):
        """Set low beta section."""
        self.energy = 3
        self.current = 100
        self.sigmaz = 2.9
        self.nat_emittance = 2.5e-10
        self.coupling_constant = 0.01
        self.energy_spread = 0.00084
        self.betax = 1.499
        self.betay = 1.435
        self.alphax = 0
        self.alphay = 0
        self.etax = 0
        self.etay = 0
        self.etapx = 0
        self.etapy = 0
        self.bsc0_h = self.bsc0_h_lowbeta
        self.bsc0_v = self.bsc0_v_lowbeta

    def set_high_beta_section(self):
        """Set high beta section."""
        self.energy = 3
        self.current = 100
        self.sigmaz = 2.9
        self.nat_emittance = 2.5e-10
        self.coupling_constant = 0.01
        self.energy_spread = 0.00084
        self.betax = 17.20
        self.betay = 3.605
        self.alphax = 0
        self.alphay = 0
        self.etax = 0
        self.etay = 0
        self.etapx = 0
        self.etapy = 0
        self.bsc0_h = self.bsc0_h_highbeta
        self.bsc0_v = self.bsc0_v_highbeta

    def calc_beam_stay_clear(self, pos):
        """Calculate horizontal and vertical BSC at a given position.

        Args:
            pos (float): position (distance from straight section center)
             in [m]

        Returns:
            float: Horizontal and Vertical beam stay clear at 'pos' [m].
        """
        beta_h = pos**2 / self.betax + self.betax
        beta_v = pos**2 / self.betay + self.betay

        bsc_h = self.bsc0_h * _np.sqrt(beta_h / self.betax)
        bsc_v = self.bsc0_v * _np.sqrt(beta_v / self.betay)

        return bsc_h, bsc_v
