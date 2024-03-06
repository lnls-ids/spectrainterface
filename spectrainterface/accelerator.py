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
            raise ValueError('Argument must be bool!')
        else:
            self._zero_emittance = value

    @zero_energy_spread.setter
    def zero_energy_spread(self, value):
        if type(value) is not bool:
            raise ValueError('Argument must be bool!')
        else:
            self._zero_energy_spread = value

    @injection_condition.setter
    def injection_condition(self, value):
        if type(value) is not str:
            raise ValueError('Argument must be str!')
        else:
            self._injection_condition = value

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


    @staticmethod
    def calc_beam_stay_clear(pos, section, delta_prototype_chamber=False):
        """Calculate horizontal and vertical beam stay clear at a given position 'pos'. 

        Args:
            pos (float): position (distance from straight section center) in [m]
            section (str): label of straight section type
            delta_prototype_chamber (bool, optional): Check if the delta prototype is being considered. Defaults to False.

        Returns:
            float, float: Horizontal and Vertical beam stay clear at 'pos'.
        """
        if section.lower() in ('sb', 'sp'):
            beta0_h = 1.499
            beta0_v = 1.435
            bsc0_h = 3.32
            bsc0_v = 1.85

            if delta_prototype_chamber:
                bsc0_h = 2.85

        elif section.lower() == 'sa':
            beta0_h = 17.20
            beta0_v = 3.605
            bsc0_h = 11.27
            bsc0_v = 2.92

            if delta_prototype_chamber:
                bsc0_h = 9.66

        beta_h = pos**2/beta0_h + beta0_h
        beta_v = pos**2/beta0_v + beta0_v

        bsc_h = bsc0_h*_np.sqrt(beta_h/beta0_h)
        bsc_v = bsc0_v*_np.sqrt(beta_v/beta0_v)

        return bsc_h, bsc_v