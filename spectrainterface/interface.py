"""Spectra functions."""

import numpy as _np
import matplotlib.pyplot as _plt
from matplotlib import colors
from spectrainterface.sirius import SIRIUS
import mathphys
from spectrainterface.tools import SourceFunctions
from spectrainterface.sources import Undulator
from scipy.interpolate import make_interp_spline
import json
from spectrainterface import spectra
import sys
import time
import copy
import os
import multiprocessing

# REPOS_PATH = os.path.abspath("./")
# REPOS_PATH = __file__
REPOS_PATH = os.path.dirname(os.path.abspath(__file__))

ECHARGE = mathphys.constants.elementary_charge
EMASS = mathphys.constants.electron_mass
LSPEED = mathphys.constants.light_speed
ECHARGE_MC = ECHARGE / (2 * _np.pi * EMASS * LSPEED)
PLANCK = mathphys.constants.reduced_planck_constant
VACUUM_PERMITTICITY = mathphys.constants.vacuum_permitticity
PI = _np.pi


class SpectraTools:
    """Class with general spectra tools."""

    @staticmethod
    def _run_solver(input_template, verbose=False):
        """Run spectra.

        Args:
            input_template (dict): Dictionary containing
            calculation parameters.
            verbose (bool): If true it will print elapsed time

        Returns:
            dict: Output data dictionary
        """
        input_str = json.dumps(input_template)

        # call solver with the input string (JSON format)
        solver = spectra.Solver(input_str)

        # check if the parameter load is OK
        isready = solver.IsReady()
        if isready is False:
            print('Parameter load failed.')
            sys.exit()

        t0 = time.time()
        # start calculation
        solver.Run()
        dt = time.time() - t0
        if verbose:
            print('elapsed time: {0:.1f} s'.format(dt))
        return solver

    @staticmethod
    def _set_accelerator_config(accelerator, input_template, flag_bend):
        input_template['Accelerator']['Energy (GeV)'] = accelerator.energy
        input_template['Accelerator']['Current (mA)'] = accelerator.current

        input_template['Accelerator']['&sigma;<sub>z</sub> (mm)'] = (
            accelerator.sigmaz
        )

        input_template['Accelerator']['Nat. Emittance (m.rad)'] = (
            accelerator.nat_emittance
        )

        input_template['Accelerator']['Coupling Constant'] = (
            accelerator.coupling_constant
        )

        input_template['Accelerator']['Energy Spread'] = (
            accelerator.energy_spread
        )

        input_template['Accelerator']['&beta;<sub>x,y</sub> (m)'] = [
            accelerator.betax,
            accelerator.betay,
        ]

        input_template['Accelerator']['&alpha;<sub>x,y</sub>'] = [
            accelerator.alphax,
            accelerator.alphay,
        ]

        input_template['Accelerator']['&eta;<sub>x,y</sub> (m)'] = [
            accelerator.etax,
            accelerator.etay,
        ]

        input_template['Accelerator']["&eta;'<sub>x,y</sub>"] = [
            accelerator.etapx,
            accelerator.etapy,
        ]

        if accelerator.injection_condition is None:
            if flag_bend:
                input_template['Accelerator']['Options'][
                    'Injection Condition'
                ] = 'Align at Center'
            else:
                input_template['Accelerator']['Options'][
                    'Injection Condition'
                ] = 'Align at Entrance'
        else:
            input_template['Accelerator']['Options']['Injection Condition'] = (
                accelerator.injection_condition
            )

        input_template['Accelerator']['Options']['Zero Emittance'] = (
            accelerator.zero_emittance
        )

        input_template['Accelerator']['Options']['Zero Energy Spread'] = (
            accelerator.zero_energy_spread
        )

        return input_template


class GeneralConfigs(SourceFunctions):
    """Class with general configs."""

    class SourceType:
        """Sub class to define source type."""

        user_defined = 'userdefined'
        horizontal_undulator = 'linearundulator'
        vertical_undulator = 'verticalundulator'
        helical_undulator = 'helicalundulator'
        elliptic_undulator = 'ellipticundulator'
        figure8_undulator = 'figure8undulator'
        vertical_figure8_undulator = 'verticalfigure8undulator'
        bending_magnet = 'bendingmagnet'
        wiggler = 'wiggler'

    class InjectionCondition:
        """Sub class to define injection condition"""

        entrance = 'Align at Entrance'
        center = 'Align at Center'
        exit = 'Align at Exit'
        automatic = 'Automatic'
        custom = 'Custom'

    def __init__(self):
        """Class constructor."""
        self._distance_from_source = 10  # [m]
        self._source_type = self.SourceType.user_defined
        self._field = None
        self._length = None
        self._bx_peak = None
        self._by_peak = None
        self._period = None
        self._kx = None
        self._ky = None
        self._rho = None

    @property
    def source_type(self):
        """Source type.

        Returns:
            CalcConfigs variables: Magnetic field, it can be defined by user or
            generated by spectra.
        """
        return self._source_type

    @property
    def field(self):
        """Magnetic field defined by user.

        Returns:
            numpy array: First column contains longitudinal spatial
            coordinate (z) [mm], second column contais vertical field
            [T], and third column constais horizontal field [T].
        """
        return self._field

    @property
    def length(self):
        """Length of device.

        Returns:
            float: Length [m]
        """
        return self._length

    @property
    def period(self):
        """Insertion device period [mm].

        Returns:
            float: ID's period [mm]
        """
        return self._period

    @property
    def by_peak(self):
        """Insertion device vertical peak field [T].

        Returns:
            float: by peak field [T]
        """
        return self._by_peak

    @property
    def bx_peak(self):
        """Insertion device horizontal peak field [T].

        Returns:
            float: bx peak field [T]
        """
        return self._bx_peak

    @property
    def ky(self):
        """Vertical deflection parameter (Ky).

        Returns:
            float: Vertical deflection parameter
        """
        return self._ky

    @property
    def kx(self):
        """Horizontal deflection parameter (Kx).

        Returns:
            float: Horizontal deflection parameter
        """
        return self._kx

    @property
    def rho(self):
        """Curvature radius.

        Returns:
            float: Curvature radius [m]
        """
        return self._rho

    @property
    def distance_from_source(self):
        """Distance from source.

        Returns:
            float: Distance from source [m]
        """
        return self._distance_from_source

    @source_type.setter
    def source_type(self, value):
        self._source_type = value

    @field.setter
    def field(self, value):
        if self.source_type != self.SourceType.user_defined:
            raise ValueError(
                'Field can only be defined if source type is user_defined.'
            )
        else:
            self._field = value

    @length.setter
    def length(self, value):
        self._length = value

    @period.setter
    def period(self, value):
        if (
            self.source_type == self.SourceType.user_defined
            or self.source_type == self.SourceType.bending_magnet
        ):
            raise ValueError(
                'Period can only be defined if source type is not user_defined or is not a bending.'  # noqa: E501
            )
        else:
            self._period = value
            if self._bx_peak is not None:
                self._kx = (
                    1e-3
                    * ECHARGE_MC
                    * self._bx_peak
                    * (
                        2 * self.period
                        if self.source_type
                        == self.SourceType.figure8_undulator
                        else self.period
                    )
                )
            if self._by_peak is not None:
                self._ky = 1e-3 * ECHARGE_MC * self._by_peak * self.period
            if self._kx is not None:
                self._bx_peak = self._kx / (
                    ECHARGE_MC
                    * 1e-3
                    * (
                        2 * self.period
                        if self.source_type
                        == self.SourceType.figure8_undulator
                        else self.period
                    )
                )
            if self._ky is not None:
                self._by_peak = self._ky / (ECHARGE_MC * 1e-3 * self.period)

    @by_peak.setter
    def by_peak(self, value):
        if self.source_type == self.SourceType.user_defined:
            raise ValueError(
                'By peak can only be defined if source type is not user_defined.'  # noqa: E501
            )
        elif self.source_type == self.SourceType.vertical_undulator:
            raise ValueError(
                'By peak can not be defined if source type is a vertical undulator.'  # noqa: E501
            )
        else:
            self._by_peak = value
            if self.period is not None:
                self._ky = 1e-3 * ECHARGE_MC * self._by_peak * self.period
                if (
                    self.source_type
                    == self.SourceType.vertical_figure8_undulator
                ):
                    self._ky *= 2

            if self.source_type == self.SourceType.helical_undulator:
                self._bx_peak = value
                if self.period is not None:
                    self._kx = self.ky

    @bx_peak.setter
    def bx_peak(self, value):
        if self.source_type == self.SourceType.user_defined:
            raise ValueError(
                'Bx peak can only be defined if source type is not user_defined.'  # noqa: E501
            )
        elif self.source_type == self.SourceType.horizontal_undulator:
            raise ValueError(
                'Bx peak can not be defined if source type is a horizontal undulator.'  # noqa: E501
            )
        else:
            self._bx_peak = value
            if self.period is not None:
                self._kx = 1e-3 * ECHARGE_MC * self._bx_peak * self.period
                if self.source_type == self.SourceType.figure8_undulator:
                    self._kx *= 2

            if self.source_type == self.SourceType.helical_undulator:
                self._by_peak = value
                if self.period is not None:
                    self._ky = self.kx

    @ky.setter
    def ky(self, value):
        if self.source_type == self.SourceType.user_defined:
            raise ValueError(
                'Ky can only be defined if source type is not user_defined.'  # noqa: E501
            )
        elif self.source_type == self.SourceType.vertical_undulator:
            raise ValueError(
                'Ky can not be defined if source type is a vertical undulator.'
            )  # noqa: E501
        else:
            self._ky = value
            if self.period is not None:
                self._by_peak = self._ky / (ECHARGE_MC * 1e-3 * self.period)

            if self.source_type == self.SourceType.helical_undulator:
                self._kx = value
                if self.period is not None:
                    self._bx_peak = self.bx_peak

    @kx.setter
    def kx(self, value):
        if self.source_type == self.SourceType.user_defined:
            raise ValueError(
                'Kx can only be defined if source type is not user_defined.'  # noqa: E501
            )
        elif self.source_type == self.SourceType.horizontal_undulator:
            raise ValueError(
                'Kx can not be defined if source type is a horizontal undulator.'  # noqa: E501
            )
        else:
            self._kx = value
            if self.period is not None:
                self._bx_peak = self._kx / (ECHARGE_MC * 1e-3 * self.period)
                if self.source_type == self.SourceType.figure8_undulator:
                    self._bx_peak /= 2

            if self.source_type == self.SourceType.helical_undulator:
                self._kx = value
                if self.period is not None:
                    self._bx_peak = self.bx_peak

    @rho.setter
    def rho(self, value):
        if self.source_type == self.SourceType.bending_magnet:
            self._rho = value
        else:
            raise ValueError(
                'Curvature radius can only be defined if source is a bending magnet.'  # noqa: E501
            )

    @distance_from_source.setter
    def distance_from_source(self, value):
        self._distance_from_source = value


class Calc(GeneralConfigs, SpectraTools):
    """Class with methods to calculate flux."""

    class CalcConfigs:
        """Sub class to define calculation parameters."""

        class Method:
            """Sub class to define calculation method."""

            fixedpoint_near_field = 'fpnearfield'
            fixedpoint_far_field = 'fpfarfield'
            near_field = 'nearfield'
            far_field = 'farfield'
            fixedpoint_wigner = 'fpwigner'
            wigner = 'wigner'

        class Variable:
            """Sub class to define independet variable."""

            energy = 'en'
            mesh_xy = 'xy'
            mesh_xxp = 'xxp'
            mesh_yyp = 'yyp'
            k = 'k'

        class Output:
            """Sub class to define output type."""

            flux_density = 'fluxdensity'
            flux = 'partialflux'
            brilliance = 'brilliance'
            power_density = 'powerdensity'
            power = 'partialpower'
            phasespace = 'phasespace'

        class SlitShape:
            """Sub class to define slit shape."""

            none = ''
            circular = 'circslit'
            rectangular = 'retslit'

    def __init__(self, accelerator):
        """Class constructor."""
        super().__init__()
        self._method = self.CalcConfigs.Method.near_field
        self._indep_var = self.CalcConfigs.Variable.energy
        self._output_type = self.CalcConfigs.Output.flux_density
        self._slit_shape = self.CalcConfigs.SlitShape.none
        self._accelerator = accelerator
        self._input_template = None

        # Energy related
        self._target_harmonic = None
        self._energy_range = None
        self._energy_step = None
        self._slit_position = None
        self._slit_acceptance = None

        #  Mesh xy related
        self._target_energy = None
        self._x_range = None
        self._y_range = None
        self._xp_range = None
        self._yp_range = None
        self._x_nr_pts = None
        self._xp_nr_pts = None
        self._y_nr_pts = None
        self._yp_nr_pts = None

        #  K related
        self._harmonic_range = None
        self._k_range = None
        self._k_nr_pts = None
        self._slice_x = None
        self._slice_y = None
        self._slice_px = None
        self._slice_py = None

        #  Phase error
        self._add_phase_errors = False
        self._use_recovery_params = True

        # Output
        self._output_captions = None
        self._output_data = None
        self._output_variables = None

        self._flux = None
        self._power_density = None
        self._power = None
        self._brilliance = None
        self._pl = None
        self._pc = None
        self._pl45 = None
        self._energies = None
        self._x = None
        self._y = None
        self._k = None
        self._output_kx = None
        self._output_ky = None
        self._dict_configs = {
            # Source Type
            'source_type': {
                'bendingmagnet': 'Bending Magnet',
                'linearundulator': 'Linear Undulator',
                'verticalundulator': 'Vertical Undulator',
                'ellipticundulator': 'Elliptic Undulator',
                'helicalundulator': 'Helical Undulator',
                'figure8undulator': 'Figure-8 Undulator',
                'wiggler': 'Wiggler',
                'userdefined': 'User Defined',
            },
            # Configurations
            'configurations': {
                # Method
                'farfield': 'Far Field & Ideal Condition',
                'fpfarfield': 'Fixed Point Calculation::Far Field & Ideal Condition',
                'nearfield': 'Near Field',
                'fpnearfield': 'Fixed Point Calculation::Near Field',
                'wigner': 'Characterization at the Source Point::Wigner Function',
                'fpwigner': 'Fixed Point Calculation::Wigner Function',
                # Variable
                'en': 'Energy Dependence',
                'k': 'K Dependence::Peak Flux Curve',  # if not wigner ::Peak Flux Curve
                'xy': 'Spatial Dependence',
                'xxp': "X-X' (Projected)",
                'yyp': "Y-Y' (Projected)",
                # Output
                'fluxdensity': 'Angular Flux Density',
                'partialflux': 'Partial Flux',
                'partialpower': 'Partial Power',
                'powerdensity': 'Spatial Power Density',  # or Angular Power Density
                'brilliance': 'Sliced',  # or Target Harmonics
                'phasespace': 'Phase-Space Distribution',
                # Slit Shape
                'circslit': 'Circular Slit',
                'retslit': 'Rectangular Slit',
            },
        }

    @property
    def method(self):
        """Method of calculation.

        Returns:
            CalcConfigs variables: Method of calculation, it can be near field
            or wigner functions, for example.
        """
        return self._method

    @property
    def indep_var(self):
        """Independent variable.

        Returns:
            CalcConfigs variables: Independet variable, it can be energy of a
            mesh in the xy plane
        """
        return self._indep_var

    @property
    def output_type(self):
        """Output type.

        Returns:
            CalcConfigs variables: Output type, it can be flux density or
            partial flux, for example.
        """
        return self._output_type

    @property
    def energy_range(self):
        """Energy range.

        Returns:
            List of ints: Energy range to calculate spectrum
             [initial point, final point].
        """
        return self._energy_range

    @property
    def energy_step(self):
        """Energy step.

        Returns:
            float: Spectrum energy step.
        """
        return self._energy_step

    @property
    def observation_angle(self):
        """Observation position [mrad].

        Returns:
            List of floats: Slit position [xpos, ypos] [mrad]
        """
        return self._slit_position

    @property
    def slit_acceptance(self):
        """Slit acceptance [mrad].

        Returns:
            List of floats: Slit acceptance [xpos, ypos] [mrad]
        """
        return self._slit_acceptance

    @property
    def slit_shape(self):
        """Slit shape.

        Returns:
            string: It can be circular or rectangular.
        """
        return self._slit_shape

    @property
    def target_energy(self):
        """Target energy.

        Returns:
            float: Target energy to analyse.
        """
        return self._target_energy

    @property
    def x_range(self):
        """Mesh x range.

        Returns:
            List of floats: x limits [mrad] [initial point, final point]
        """
        return self._x_range

    @property
    def xp_range(self):
        """Mesh x' range.

        Returns:
            List of floats: x' limits [mrad] [initial point, final point]
        """
        return self._xp_range

    @property
    def y_range(self):
        """Mesh y range.

        Returns:
            List of floats: y limits [mrad] [initial point, final point]
        """
        return self._y_range

    @property
    def yp_range(self):
        """Mesh y' range.

        Returns:
            List of floats: y' limits [mrad] [initial point, final point]
        """
        return self._yp_range

    @property
    def x_nr_pts(self):
        """Nr of x points.

        Returns:
            float: Number of horizontal mesh points
        """
        return self._x_nr_pts

    @property
    def xp_nr_pts(self):
        """Nr of x' points.

        Returns:
            float: Number of horizontal angle mesh points
        """
        return self._xp_nr_pts

    @property
    def y_nr_pts(self):
        """Nr of y points.

        Returns:
            float: Number of vertical mesh points
        """
        return self._y_nr_pts

    @property
    def yp_nr_pts(self):
        """Nr of y' points.

        Returns:
            float: Number of vertical angle mesh points
        """
        return self._yp_nr_pts

    @property
    def harmonic_range(self):
        """Harmonic range.

        Returns:
            list of ints: List of harmonics to calculate brilliance.
        """
        return self._harmonic_range

    @property
    def target_harmonic(self):
        """Harmonic number.

        Returns:
            int: number of harmonic to calculate brilliance.
        """
        return self._target_harmonic

    @property
    def k_range(self):
        """K range.

        Returns:
            list of ints: List of k to calculate brilliance [kmin, kmax].
        """
        return self._k_range

    @property
    def k_nr_pts(self):
        """Number of k points.

        Returns:
            int: Number of K points.
        """
        return self._k_nr_pts

    @property
    def slice_x(self):
        """Slice x.

        Returns:
            float: Horizontal source point where Wigner function is
             calculated [mm].
        """
        return self._slice_x

    @property
    def slice_y(self):
        """Slice y.

        Returns:
            float: Vertical source point where Wigner function is
             calculated [mm].
        """
        return self._slice_y

    @property
    def slice_px(self):
        """Slice x'.

        Returns:
            float: Horizontal source angle where Wigner function is
             calculated [mrad].
        """
        return self._slice_px

    @property
    def slice_py(self):
        """Slice y'.

        Returns:
            float: Horizontal source angle where Wigner function is
             calculated [mrad].
        """
        return self._slice_py

    @property
    def output_captions(self):
        """Output captions.

        Returns:
            dict: Captions with spectra output
        """
        return self._output_captions

    @property
    def output_data(self):
        """Output data.

        Returns:
            dict: Data output from spectra
        """
        return self._output_data

    @property
    def output_variables(self):
        """Output variables.

        Returns:
            dict: Variables from spectra
        """
        return self._output_variables

    @property
    def flux(self):
        """Flux output.

        Returns:
            numpy array: Flux [ph/s/mr²/0.1%B.W].
        """
        return self._flux

    @property
    def power_density(self):
        """Power density output.

        Returns:
            numpy array: power density [kW/mr²].
        """
        return self._power_density

    @property
    def power(self):
        """Partial power output.

        Returns:
            numpy array: power density [kW].
        """
        return self._power

    @property
    def brilliance(self):
        """Brilliance output.

        Returns:
            numpy array: Brilliance [ph/s/mr²/0.1%B.W/mm²].
        """
        return self._brilliance

    @property
    def pl(self):
        """Linear polarization.

        Returns:
            numpy array: Linear polarization s1/s0
        """
        return self._pl

    @property
    def pc(self):
        """Circular polarization.

        Returns:
            numpy array: Circular polarization s3/s0
        """
        return self._pc

    @property
    def pl45(self):
        """Linear polarization 45°.

        Returns:
            numpy array: Linear polarization 45° s2/s0
        """
        return self._pl45

    @property
    def energies(self):
        """Energies.

        Returns:
            numpy array: Energyes [eV]
        """
        return self._energies

    @property
    def x(self):
        """Horizontal angle.

        Returns:
            numpy array: Horizontal angle [mrad]
        """
        return self._x

    @property
    def y(self):
        """Vertical angle.

        Returns:
            numpy array: Vertical angle [mrad]
        """
        return self._y

    @property
    def k(self):
        """Deflection parameter K.

        Returns:
            numpy array: Deflecetion parameter K.
        """
        return self._k

    @property
    def output_kx(self):
        """Deflection parameter Kx.

        Returns:
            numpy array: Deflecetion parameter Kx.
        """
        return self._kx

    @property
    def output_ky(self):
        """Deflection parameter Ky.

        Returns:
            numpy array: Deflecetion parameter Ky.
        """
        return self._ky

    @method.setter
    def method(self, value):
        self._method = value

    @indep_var.setter
    def indep_var(self, value):
        self._indep_var = value
        if value == self.CalcConfigs.Variable.energy:
            self._slit_position = [0, 0]
        elif value == self.CalcConfigs.Variable.mesh_xy:
            self._slit_position = None
            self._slit_shape = self.CalcConfigs.SlitShape.none

    @output_type.setter
    def output_type(self, value):
        self._output_type = value
        if value == self.CalcConfigs.Output.flux_density:
            self._slit_shape = self.CalcConfigs.SlitShape.none

    @energy_range.setter
    def energy_range(self, value):
        if self.indep_var != self.CalcConfigs.Variable.energy:
            raise ValueError(
                'Energy range can only be defined if the independent variable is energy.'  # noqa: E501
            )
        else:
            self._energy_range = value

    @energy_step.setter
    def energy_step(self, value):
        if self.indep_var != self.CalcConfigs.Variable.energy:
            raise ValueError(
                'Energy step can only be defined if the independent variable is energy.'  # noqa: E501
            )
        else:
            self._energy_step = value

    @observation_angle.setter
    def observation_angle(self, value):
        if self.indep_var != self.CalcConfigs.Variable.energy:
            raise ValueError(
                'Observation position can only be defined if the independent variable is energy.'  # noqa: E501
            )
        else:
            self._slit_position = value

    @slit_acceptance.setter
    def slit_acceptance(self, value):
        if self.output_type == self.CalcConfigs.Output.power:
            self._slit_acceptance = value
        elif self.output_type != self.CalcConfigs.Output.flux:
            if self.source_type == self.SourceType.bending_magnet:
                self._slit_acceptance = value
            else:
                raise ValueError(
                    'Slit acceptance can only be defined if the output type is flux.'  # noqa: E501
                )
        else:
            self._slit_acceptance = value

    @slit_shape.setter
    def slit_shape(self, value):
        if (
            self.output_type == self.CalcConfigs.Output.flux
            or self.output_type == self.CalcConfigs.Output.power
        ):
            self._slit_shape = value
        else:
            raise ValueError(
                'Slit shape can only be defined if the output type is flux.'  # noqa: E501
            )

    @target_energy.setter
    def target_energy(self, value):
        if self.indep_var != self.CalcConfigs.Variable.mesh_xy:
            if self.indep_var == self.CalcConfigs.Variable.energy:
                if (
                    self.method
                    == self.CalcConfigs.Method.fixedpoint_near_field
                ):
                    self._target_energy = value
                elif (
                    self.method == self.CalcConfigs.Method.fixedpoint_far_field
                ):
                    self._target_energy = value
                else:
                    raise ValueError(
                        'Target energy can only be defined if the variable is a xy mesh or calculation method is a fixed point.'  # noqa: E501
                    )
            else:
                raise ValueError(
                    'Target energy can only be defined if the variable is a xy mesh or calculation method is a fixed point.'  # noqa: E501
                )
        else:
            self._target_energy = value

    @x_range.setter
    def x_range(self, value):
        if (
            self.indep_var == self.CalcConfigs.Variable.mesh_xxp
            and self.output_type == self.CalcConfigs.Output.phasespace
            and self.method == self.CalcConfigs.Method.wigner
        ):
            self._x_range = value
        elif self.indep_var == self.CalcConfigs.Variable.mesh_xy:
            self._x_range = value
        else:
            raise ValueError(
                "X range can only be defined if the variable is a xy mesh or xx' phasespace."  # noqa: E501
            )

    @xp_range.setter
    def xp_range(self, value):
        if (
            self.indep_var == self.CalcConfigs.Variable.mesh_xxp
            and self.output_type == self.CalcConfigs.Output.phasespace
            and self.method == self.CalcConfigs.Method.wigner
        ):
            self._xp_range = value
        elif self.indep_var == self.CalcConfigs.Variable.mesh_xy:
            self._xp_range = value
        else:
            raise ValueError(
                "X range can only be defined if the variable is a xy mesh or xx' phasespace."  # noqa: E501
            )

    @y_range.setter
    def y_range(self, value):
        if (
            self.indep_var == self.CalcConfigs.Variable.mesh_yyp
            and self.output_type == self.CalcConfigs.Output.phasespace
            and self.method == self.CalcConfigs.Method.wigner
        ):
            self._y_range = value
        elif self.indep_var == self.CalcConfigs.Variable.mesh_xy:
            self._y_range = value
        else:
            raise ValueError(
                "X range can only be defined if the variable is a xy mesh or yy' phasespace."  # noqa: E501
            )

    @yp_range.setter
    def yp_range(self, value):
        if (
            self.indep_var == self.CalcConfigs.Variable.mesh_yyp
            and self.output_type == self.CalcConfigs.Output.phasespace
            and self.method == self.CalcConfigs.Method.wigner
        ):
            self._yp_range = value
        elif self.indep_var == self.CalcConfigs.Variable.mesh_xy:
            self._yp_range = value
        else:
            raise ValueError(
                "X range can only be defined if the variable is a xy mesh or or yy' phasespace."  # noqa: E501
            )

    @x_nr_pts.setter
    def x_nr_pts(self, value):
        if (
            self.indep_var == self.CalcConfigs.Variable.mesh_xxp
            and self.output_type == self.CalcConfigs.Output.phasespace
            and self.method == self.CalcConfigs.Method.wigner
        ):
            self._x_nr_pts = value
        elif self.indep_var == self.CalcConfigs.Variable.mesh_xy:
            self._x_nr_pts = value
        else:
            raise ValueError(
                "X range can only be defined if the variable is a xy mesh or xx' phasespace."  # noqa: E501
            )

    @xp_nr_pts.setter
    def xp_nr_pts(self, value):
        if (
            self.indep_var == self.CalcConfigs.Variable.mesh_xxp
            and self.output_type == self.CalcConfigs.Output.phasespace
            and self.method == self.CalcConfigs.Method.wigner
        ):
            self._xp_nr_pts = value
        else:
            raise ValueError(
                "X range can only be defined if the variable is a or xx' phasespace."  # noqa: E501
            )

    @y_nr_pts.setter
    def y_nr_pts(self, value):
        if (
            self.indep_var == self.CalcConfigs.Variable.mesh_yyp
            and self.output_type == self.CalcConfigs.Output.phasespace
            and self.method == self.CalcConfigs.Method.wigner
        ):
            self._y_nr_pts = value
        elif self.indep_var == self.CalcConfigs.Variable.mesh_xy:
            self._y_nr_pts = value
        else:
            raise ValueError(
                'Y range can only be defined if the variable is a xy mesh.'  # noqa: E501
            )

    @yp_nr_pts.setter
    def yp_nr_pts(self, value):
        if (
            self.indep_var == self.CalcConfigs.Variable.mesh_yyp
            and self.output_type == self.CalcConfigs.Output.phasespace
            and self.method == self.CalcConfigs.Method.wigner
        ):
            self._yp_nr_pts = value
        else:
            raise ValueError(
                "X range can only be defined if the variable is a or yy' phasespace."  # noqa: E501
            )

    @harmonic_range.setter
    def harmonic_range(self, value):
        if self.indep_var != self.CalcConfigs.Variable.k:
            raise ValueError(
                'Harmonic range can only be defined if the variable is k.'
            )
        else:
            self._harmonic_range = value

    @target_harmonic.setter
    def target_harmonic(self, value):
        if (
            self.indep_var == self.CalcConfigs.Variable.mesh_xxp
            or self.indep_var == self.CalcConfigs.Variable.mesh_yyp
        ) and self.output_type == self.CalcConfigs.Output.phasespace:
            self._target_harmonic = value
        elif self.indep_var == self.CalcConfigs.Variable.energy:
            if self.method == self.CalcConfigs.Method.fixedpoint_wigner:
                self._target_harmonic = value
            else:
                raise ValueError(
                    'Harmonic number can only be defined if the method is fixed point wigner.'  # noqa: E501
                )
        else:
            raise ValueError(
                'Harmonic number can only be defined if the variable is energy.'  # noqa: E501
            )

    @k_range.setter
    def k_range(self, value):
        if self.indep_var != self.CalcConfigs.Variable.k:
            raise ValueError(
                'K range can only be defined if the variable is k.'
            )
        else:
            self._k_range = value

    @k_nr_pts.setter
    def k_nr_pts(self, value):
        if self.indep_var != self.CalcConfigs.Variable.k:
            raise ValueError(
                'K nr points can only be defined if the variable is k.'
            )
        else:
            self._k_nr_pts = value

    @slice_x.setter
    def slice_x(self, value):
        if self.indep_var == self.CalcConfigs.Variable.k:
            self._slice_x = value
        elif self.indep_var == self.CalcConfigs.Variable.energy:
            if self.method == self.CalcConfigs.Method.fixedpoint_wigner:
                self._slice_x = value
            else:
                raise ValueError(
                    'Slice x can only be defined if the method is fixed point wigner.'  # noqa: E501
                )
        else:
            raise ValueError(
                'Slice x can only be defined if the variable is k  or energy.'
            )

    @slice_y.setter
    def slice_y(self, value):
        if self.indep_var == self.CalcConfigs.Variable.k:
            self._slice_y = value
        elif self.indep_var == self.CalcConfigs.Variable.energy:
            if self.method == self.CalcConfigs.Method.fixedpoint_wigner:
                self._slice_y = value
            else:
                raise ValueError(
                    'Slice y can only be defined if the method is fixed point wigner.'  # noqa: E501
                )
        else:
            raise ValueError(
                'Slice y can only be defined if the variable is k  or energy.'
            )

    @slice_px.setter
    def slice_px(self, value):
        if self.indep_var == self.CalcConfigs.Variable.k:
            self._slice_px = value
        elif self.indep_var == self.CalcConfigs.Variable.energy:
            if self.method == self.CalcConfigs.Method.fixedpoint_wigner:
                self._slice_px = value
            else:
                raise ValueError(
                    "Slice x' can only be defined if the method is fixed point wigner."  # noqa: E501
                )
        else:
            raise ValueError(
                "Slice x' can only be defined if the variable is k  or energy."
            )

    @slice_py.setter
    def slice_py(self, value):
        if self.indep_var == self.CalcConfigs.Variable.k:
            self._slice_py = value
        elif self.indep_var == self.CalcConfigs.Variable.energy:
            if self.method == self.CalcConfigs.Method.fixedpoint_wigner:
                self._slice_py = value
            else:
                raise ValueError(
                    "Slice y' can only be defined if the method is fixed point wigner."  # noqa: E501
                )
        else:
            raise ValueError(
                "Slice y' can only be defined if the variable is k or energy."
            )

    def _reset_class(self):
        """Reset Class Function."""
        # Energy related
        self._target_harmonic = None
        self._energy_range = None
        self._energy_step = None
        self._slit_position = None
        self._slit_acceptance = None

        #  Mesh xy related
        self._target_energy = None
        self._x_range = None
        self._y_range = None
        self._x_nr_pts = None
        self._y_nr_pts = None

        #  K related
        self._harmonic_range = None
        self._k_range = None
        self._k_nr_pts = None
        self._slice_x = None
        self._slice_y = None
        self._slice_px = None
        self._slice_py = None

    def set_config(self):  # noqa: C901
        """Set calc config."""
        config_name = self.source_type
        config_name += '_'
        config_name += self.method
        config_name += '_'
        config_name += self.indep_var
        config_name += '_'
        config_name += self.output_type

        if self.slit_shape != '':
            config_name += '_'
            config_name += self.slit_shape

        # Assembly of source and configuration type
        keys = config_name.split('_')

        source_type = self._dict_configs['source_type'][keys[0]]

        config = [
            '::' + self._dict_configs['configurations'][key]
            if key in self._dict_configs['configurations']
            else ''
            for key in keys[1::]
        ]
        config[0] = config[0][2::]
        config_type = ''.join(config)
        config_type = (
            config_type.replace('Energy Dependence::', '')
            if ('fpfarfield' in keys)
            or ('fpnearfield' in keys)
            or ('fpwigner' in keys)
            else config_type
        )
        config_type = (
            config_type.replace('::Circular Slit', '')
            if ('fpwigner' in keys)
            else config_type
        )
        config_type = (
            config_type.replace('::Rectangular Slit', '')
            if ('fpwigner' in keys)
            else config_type
        )
        config_type = (
            config_type.replace(
                'Angular Flux Density', 'Spatial Flux Density::Mesh: x-y'
            )
            if 'xy' in keys
            else config_type
        )
        config_type = (
            config_type.replace(
                'Spatial Power Density', 'Spatial Power Density::Mesh: x-y'
            )
            if 'xy' in keys
            else config_type
        )
        config_type = (
            config_type + '::Target Harmonics' if 'k' in keys else config_type
        )
        config_type = (
            config_type.replace('Spatial Flux Density', 'Angular Flux Density')
            if 'farfield' in keys
            else config_type
        )
        config_type = (
            config_type.replace(
                'Spatial Power Density', 'Angular Power Density'
            )
            if 'farfield' in keys
            else config_type
        )
        config_type = (
            config_type.replace('Angular Flux Density', 'Spatial Flux Density')
            if 'nearfield' in keys
            else config_type
        )
        config_type = (
            config_type.replace(
                'Angular Power Density', 'Spatial Power Density'
            )
            if 'nearfield' in keys
            else config_type
        )
        config_type = (
            config_type.replace('Peak Flux Curve::Sliced::', '')
            if 'wigner' in keys
            else config_type
        )
        config_type = (
            config_type.replace(
                "Y-Y' (Projected)::Phase-Space Distribution",
                "Phase-Space Distribution::Y-Y' (Projected)",
            )
            if 'phasespace' in keys
            else config_type
        )
        config_type = (
            config_type.replace(
                "X-X' (Projected)::Phase-Space Distribution",
                "Phase-Space Distribution::X-X' (Projected)",
            )
            if 'phasespace' in keys
            else config_type
        )

        # Open template file
        template_file_name = (
            REPOS_PATH + '/calculation_parameters/parameters_template.json'
        )
        file = open(template_file_name)
        input_temp = json.load(file)

        # Setting accelerator parameters
        flag_bend = False
        if self.source_type == self.SourceType.bending_magnet:
            flag_bend = True

        input_temp = self._set_accelerator_config(
            self._accelerator, input_temp, flag_bend
        )

        # Setting configuration and source type
        input_temp['Configurations']['Type'] = config_type
        input_temp['Light Source']['Type'] = source_type

        if self.field is not None:
            data = _np.zeros((3, len(self.field[:, 0])))
            data[0, :] = self.field[:, 0]
            data[1, :] = self.field[:, 2]
            data[2, :] = self.field[:, 1]
            input_temp['Light Source']['Field Profile']['data'] = data.tolist()

        if self.ky is not None:
            if (
                self.source_type == self.SourceType.horizontal_undulator
                or self.source_type == self.SourceType.helical_undulator
                or self.source_type == self.SourceType.wiggler
            ):
                input_temp['Light Source']['K value'] = self.ky

        if self.kx is not None:
            if self.source_type == self.SourceType.vertical_undulator:
                input_temp['Light Source']['K value'] = self.kx

        if self.kx is not None or self.ky is not None:
            if (
                self.source_type == self.SourceType.elliptic_undulator
                or self.source_type == self.SourceType.figure8_undulator
                or self.source_type
                == self.SourceType.vertical_figure8_undulator
            ):
                input_temp['Light Source']['K<sub>x,y</sub>'] = [
                    self.kx if self.kx is not None else 0,
                    self.ky if self.ky is not None else 0,
                ]

        if self.by_peak is not None:
            if self.source_type == self.SourceType.bending_magnet:
                input_temp['Light Source']['B (T)'] = self.by_peak
                energy = self._accelerator.energy
                brho = energy * 1e9 * ECHARGE / (ECHARGE * LSPEED)
                rho = brho / self.by_peak
                input_temp['Light Source']['&rho; (m)'] = rho

            elif self.source_type == self.SourceType.wiggler:
                input_temp['Light Source']['B (T)'] = self.by_peak

        if self.rho is not None:
            if self.source_type == self.SourceType.bending_magnet:
                input_temp['Light Source']['&rho; (m)'] = self.rho
                energy = self._accelerator.energy
                brho = energy * 1e9 * ECHARGE / (ECHARGE * LSPEED)
                by = brho / self.rho
                input_temp['Light Source']['B (T)'] = by

        if self.period is not None:
            input_temp['Light Source']['&lambda;<sub>u</sub> (mm)'] = (
                self.period
            )

        if self.length is not None:
            if self.source_type == self.SourceType.bending_magnet:
                input_temp['Light Source']['BM Length (m)'] = self.length
            else:
                input_temp['Light Source']['Device Length (m)'] = self.length

        if self.energy_range is not None:
            input_temp['Configurations']['Energy Range (eV)'] = (
                self.energy_range
            )

        if self.energy_step is not None:
            if (
                self.source_type == self.SourceType.bending_magnet
                or self.source_type == self.SourceType.wiggler
            ):
                nr_points = int(
                    (self.energy_range[1] - self.energy_range[0])
                    / self.energy_step
                )
                input_temp['Configurations']['Points (Energy)'] = nr_points
            else:
                input_temp['Configurations']['Energy Pitch (eV)'] = (
                    self.energy_step
                )

        if self.observation_angle is not None:
            if self.output_type == self.CalcConfigs.Output.flux_density:
                input_temp['Configurations'][
                    'Angle &theta;<sub>x,y</sub> (mrad)'
                ] = self.observation_angle
            elif self.output_type == self.CalcConfigs.Output.flux:
                input_temp['Configurations'][
                    'Slit Pos.: &theta;<sub>x,y</sub> (mrad)'
                ] = self.observation_angle
            elif self.output_type == self.CalcConfigs.Output.power:
                input_temp['Configurations'][
                    'Slit Pos.: &theta;<sub>x,y</sub> (mrad)'
                ] = self.observation_angle

        if self.slit_acceptance is not None:
            if self.slit_shape == self.CalcConfigs.SlitShape.circular:
                input_temp['Configurations'][
                    'Slit &theta;<sub>1,2</sub> (mrad)'
                ] = self.slit_acceptance
            elif self.slit_shape == self.CalcConfigs.SlitShape.rectangular:
                input_temp['Configurations'][
                    '&Delta;&theta;<sub>x,y</sub> (mrad)'
                ] = self.slit_acceptance
            if self.source_type == self.SourceType.bending_magnet:
                if self.output_type == self.CalcConfigs.Output.flux_density:
                    input_temp['Configurations']["X' Acceptance (mrad)"] = (
                        self.slit_acceptance
                    )

        if self.target_energy is not None:
            input_temp['Configurations']['Target Energy (eV)'] = (
                self.target_energy
            )

        if self.target_harmonic is not None:
            input_temp['Configurations']['Target Harmonic'] = (
                self.target_harmonic
            )

        if self.x_range is not None:
            if self.output_type == self.CalcConfigs.Output.phasespace:
                input_temp['Configurations']['X Range (mm)'] = self.x_range
                input_temp['Configurations']['Points (X)'] = self.x_nr_pts
                input_temp['Configurations']["X' Range (mrad)"] = self.xp_range
                input_temp['Configurations']["Points (X')"] = self.xp_nr_pts
            else:
                input_temp['Configurations'][
                    '&theta;<sub>x</sub> Range (mrad)'
                ] = self.x_range
                input_temp['Configurations']['Points (x)'] = self.x_nr_pts

        if self.y_range is not None:
            if self.output_type == self.CalcConfigs.Output.phasespace:
                input_temp['Configurations']['Y Range (mm)'] = self.y_range
                input_temp['Configurations']['Points (Y)'] = self.y_nr_pts
                input_temp['Configurations']["Y' Range (mrad)"] = self.yp_range
                input_temp['Configurations']["Points (Y')"] = self.yp_nr_pts
            else:
                input_temp['Configurations'][
                    '&theta;<sub>y</sub> Range (mrad)'
                ] = self.y_range
                input_temp['Configurations']['Points (y)'] = self.y_nr_pts

        if self.harmonic_range is not None:
            input_temp['Configurations']['Harmonic Range'] = (
                self.harmonic_range
            )
            if (
                self.source_type == self.SourceType.horizontal_undulator
                or self.source_type == self.SourceType.vertical_undulator
                or self.source_type == self.SourceType.helical_undulator
            ):
                input_temp['Configurations']['K Range'] = self.k_range
            else:
                input_temp['Configurations']['K<sub>&perp;</sub> Range'] = (
                    self.k_range
                )
            if self.method == self.CalcConfigs.Method.wigner:
                input_temp['Configurations']['Slice X (mm)'] = self.slice_x
                input_temp['Configurations']['Slice Y (mm)'] = self.slice_y
                input_temp['Configurations']["Slice X' (mrad)"] = self.slice_px
                input_temp['Configurations']["Slice Y' (mrad)"] = self.slice_py
            input_temp['Configurations']['Points (K)'] = self.k_nr_pts

        input_temp['Configurations']['Distance from the Source (m)'] = (
            self.distance_from_source
        )

        self._input_template = input_temp

    def verify_valid_parameters(self):  # noqa: C901
        """Check if calculation parameters are valid.

        Returns:
            bool: Returns True of False.
        """
        if self.indep_var == self.CalcConfigs.Variable.energy:
            if self.target_energy is None:
                if self.energy_range is None:
                    if self.target_harmonic is None:
                        raise ValueError('Energy range must be defined.')

            if self.target_energy is None:
                if self.energy_step is None:
                    if self.target_harmonic is None:
                        raise ValueError('Energy step must be defined.')

            if self.observation_angle is None:
                raise ValueError('Observation angle must be defined.')

            if self.source_type == self.SourceType.bending_magnet:
                if self.slit_acceptance is None:
                    raise ValueError('Slit acceptance must be defined.')

            if self.output_type == self.CalcConfigs.Output.flux:
                if self.slit_acceptance is None:
                    raise ValueError('Slit acceptance must be defined.')

                if self.slit_shape is None:
                    raise ValueError('Slit shape must be defined.')

        if self.indep_var == self.CalcConfigs.Variable.mesh_xy:
            if (
                self.target_energy is None
                and self.output_type == self.CalcConfigs.Output.flux_density
            ):
                raise ValueError('Energy target must be defined')

            if self.x_range is None:
                raise ValueError('X range must be defined.')

            if self.y_range is None:
                raise ValueError('Y range must be defined.')

            if self.x_nr_pts is None:
                raise ValueError('Nr. of x points must be defined.')

            if self.y_nr_pts is None:
                raise ValueError('Nr. of y points must be defined.')

        if self.indep_var == self.CalcConfigs.Variable.mesh_xxp:
            if self.target_harmonic is None:
                raise ValueError('Harmonic number must be defined.')
            if self.x_range is not None:
                if self.xp_range is None:
                    raise ValueError("X' range must be defined.")
                if self.x_nr_pts is None:
                    raise ValueError('Nr. of x points must be defined.')
                if self.xp_nr_pts is None:
                    raise ValueError("Nr. of x' points must be defined.")
            elif self.y_range is not None:
                if self.yp_range is None:
                    raise ValueError("Y' range must be defined.")
                if self.y_nr_pts is None:
                    raise ValueError('Nr. of y points must be defined.')
                if self.yp_nr_pts is None:
                    raise ValueError("Nr. of y' points must be defined.")
            else:
                raise ValueError('X or Y range must be defined.')

        if self.indep_var == self.CalcConfigs.Variable.k:
            if self.harmonic_range is None:
                raise ValueError('Harmonic range must be defined.')

            if self.k_range is None:
                raise ValueError('K range must be defined.')

            if self.k_nr_pts is None:
                raise ValueError('Number of K points must be defined.')

            if self.method == self.CalcConfigs.Method.wigner:
                if self.slice_x is None:
                    raise ValueError('Slice x must be defined.')

                if self.slice_y is None:
                    raise ValueError('Slice y must be defined.')

                if self.slice_px is None:
                    raise ValueError('Slice px must be defined.')

                if self.slice_py is None:
                    raise ValueError('Slice py must be defined.')

        return True

    def run_calculation(self, time_print: bool = False):
        """Run calculation."""
        self.verify_valid_parameters()
        solver = self._run_solver(self._input_template, time_print)
        self._solver = solver
        captions, data, variables = self.extractdata(solver)
        self._output_captions = captions
        self._output_data = data
        self._output_variables = variables
        # self._reset_class()
        self._set_outputs()

        return solver

    def _set_outputs(self):  # noqa: C901
        data = self._output_data
        captions = self._output_captions
        variables = self._output_variables

        if self.indep_var == self.CalcConfigs.Variable.energy:
            if (
                self.method == self.CalcConfigs.Method.fixedpoint_near_field
                or self.method == self.CalcConfigs.Method.fixedpoint_far_field
            ):
                if self.output_type == self.CalcConfigs.Output.power:
                    self._power = data[0][0]
                else:
                    self._flux = data[0]
                    self._pl = data[1]
                    self._pc = data[2]
                    self._pl45 = data[3]
            elif self.method == self.CalcConfigs.Method.fixedpoint_wigner:
                self._brilliance = data[0]
            else:
                self._flux = data[0, :]
                if len(captions['titles']) == 5:
                    self._pl = data[1, :]
                    self._pc = data[2, :]
                    self._pl45 = data[3, :]
                elif len(captions['titles']) == 6:
                    self._brilliance = data[1, :]
                    self._pl = data[2, :]
                    self._pc = data[3, :]
                    self._pl45 = data[4, :]
                self._energies = self._output_variables[0, :]

        elif self.indep_var == self.CalcConfigs.Variable.mesh_xy:
            if self.output_type == self.CalcConfigs.Output.power_density:
                self._power_density = data[0, :]
                self._x = _np.array(self._output_variables[0][:])
                self._y = _np.array(self._output_variables[1][:])
                self._power_density = _np.reshape(
                    self._power_density, (len(self._y), len(self._x))
                )
                self._power_density = _np.flip(self._power_density, axis=0)

            if self.output_type == self.CalcConfigs.Output.flux_density:
                self._flux = data[0, :]
                self._x = _np.array(self._output_variables[0][:])
                self._y = _np.array(self._output_variables[1][:])
                self._flux = _np.reshape(
                    self._flux, (len(self._y), len(self._x))
                )
                self._flux = _np.flip(self._flux, axis=0)
                self._pl = data[1, :]
                self._pc = data[2, :]
                self._pl45 = data[3, :]

                self._pl = _np.reshape(self._pl, (len(self._y), len(self._x)))
                self._pl = _np.flip(self._pl, axis=0)

                self._pc = _np.reshape(self._pc, (len(self._y), len(self._x)))
                self._pc = _np.flip(self._pc, axis=0)

                self._pl45 = _np.reshape(
                    self._pl45, (len(self._y), len(self._x))
                )
                self._pl45 = _np.flip(self._pl45, axis=0)

        elif (
            self.indep_var == self.CalcConfigs.Variable.mesh_xxp
            or self.indep_var == self.CalcConfigs.Variable.mesh_yyp
        ):
            if (
                self.method == self.CalcConfigs.Method.wigner
                and self.output_type == self.CalcConfigs.Output.phasespace
            ):
                self._brilliance = data[0]

        elif self.indep_var == self.CalcConfigs.Variable.k:
            if self.method == self.CalcConfigs.Method.wigner:
                if (
                    self.source_type != self.SourceType.elliptic_undulator
                    and self.source_type != self.SourceType.figure8_undulator
                    and self.source_type
                    != self.SourceType.vertical_figure8_undulator
                ):
                    self._k = data[:, 0, :]
                    self._brilliance = data[:, 1, :]
                    self._energies = variables[:, :]
                    if self._add_phase_errors is True:
                        self._brilliance = self.apply_phase_errors(
                            self._brilliance, self._use_recovery_params
                        )

                else:
                    self._kx = data[:, 0, :]
                    self._ky = data[:, 1, :]
                    self._brilliance = data[:, 2, :]
                    self._energies = variables[:, :]
            elif self.method == self.CalcConfigs.Method.far_field:
                if (
                    self.source_type != self.SourceType.elliptic_undulator
                    and self.source_type != self.SourceType.figure8_undulator
                    and self.source_type
                    != self.SourceType.vertical_figure8_undulator
                ):
                    self._k = data[:, 1, :]
                    self._flux = data[:, 2, :]
                    self._energies = variables[:, :]
                    if self._add_phase_errors is True:
                        self._flux = self.apply_phase_errors(
                            self._flux, self._use_recovery_params
                        )
                    self._brilliance = data[:, 2, :]
                    self._pl = data[:, 3, :]
                    self._pc = data[:, 4, :]
                    self._pl45 = data[:, 5, :]
                else:
                    self._kx = data[:, 1, :]
                    self._ky = data[:, 2, :]
                    self._flux = data[:, 3, :]
                    self._energies = variables[:, :]

    def extractdata(self, solver):
        """Extract solver data.

        Args:
            solver (spectra solver): Spectra solver object

        Returns:
            dict: captions
            dict: data
            dict: variables
        """
        captions = solver.GetCaptions()
        data = _np.array(solver.GetData()['data'])

        if self.indep_var != self.CalcConfigs.Variable.k:
            variables = _np.array(solver.GetData()['variables'], dtype=object)
        else:
            if (
                self.source_type == self.SourceType.figure8_undulator
                or self.source_type
                == self.SourceType.vertical_figure8_undulator
            ):
                nr_harmonics = (
                    int((self.harmonic_range[-1] - self.harmonic_range[0]) * 2)
                    + 1
                )
                if self.method == self.CalcConfigs.Method.wigner:
                    data = _np.zeros((nr_harmonics, 3, self.k_nr_pts))
                else:
                    data = _np.zeros((nr_harmonics, 7, self.k_nr_pts))
            else:
                nr_harmonics = (
                    int((self.harmonic_range[-1] - self.harmonic_range[0]) / 2)
                    + 1
                )
                if self.source_type == self.SourceType.elliptic_undulator:
                    if self.method == self.CalcConfigs.Method.wigner:
                        data = _np.zeros((nr_harmonics, 3, self.k_nr_pts))
                    else:
                        data = _np.zeros((nr_harmonics, 7, self.k_nr_pts))
                else:
                    if self.method == self.CalcConfigs.Method.wigner:
                        data = _np.zeros((nr_harmonics, 2, self.k_nr_pts))
                    else:
                        data = _np.zeros((nr_harmonics, 6, self.k_nr_pts))
            variables = _np.zeros((nr_harmonics, self.k_nr_pts))
            for i in range(nr_harmonics):
                variables[i, :] = _np.array(
                    solver.GetDetailData(i)['variables']
                )
                data[i, :, :] = _np.array(solver.GetDetailData(i)['data'])
        return captions, data, variables

    def apply_phase_errors(self, values, rec_param=True):
        """Add phase errors.

        Args:
            values (numpy 2d array): It can be brilliance of flux
            rec_param (bool, optional): Use recovery params. Defaults to True.

        Returns:
            numpy 2d array: brilliance of flux with phase errors
        """
        fname = REPOS_PATH + '/files/phase_errors_fit.txt'
        data = _np.genfromtxt(fname, unpack=True, skip_header=1)
        # h = data[:, 0]
        ph_err1 = data[:, 1]
        ph_err2 = data[:, 2]
        harm0 = self.harmonic_range[0]
        harmf = self.harmonic_range[-1]
        idcs = (_np.arange(harm0 - 1, harmf, 2),)
        if rec_param:
            ph = ph_err2[idcs]
        else:
            ph = ph_err1[idcs]
        ph_full = _np.tile(ph, values.shape[1]).reshape(
            values.shape, order='F'
        )
        return values * ph_full

    @staticmethod
    def process_brilliance_curve(
        input_energies, input_brilliance, superp_value=250
    ):
        """Process brilliance curve.

        Args:
            input_energies (numpy 2d array): Array with energies and
             brilliances for each harmonic
            input_brilliance (numpy 2d array): Array with energies and
                brilliances for each harmonic
            superp_value (int, optional): Desired value of energy
             superposition. Defaults to 250.

        Returns:
            _type_: _description_
        """
        flag_pre_processing = False
        harm_nr = input_energies.shape[0]
        energies = _np.zeros((harm_nr, 2001))
        brilliance = _np.zeros((harm_nr, 2001))
        for i in _np.arange(harm_nr - 1):
            if flag_pre_processing is False:
                e_harm = input_energies[i, :]
                b_harm = input_brilliance[i, :]
                idx = _np.argsort(e_harm)
                e_harm = e_harm[idx]
                b_harm = b_harm[idx]
                e_harm_interp = _np.linspace(
                    _np.min(e_harm), _np.max(e_harm), 2001
                )
                b_harm_interp = _np.exp(
                    _np.interp(e_harm_interp, e_harm, _np.log(b_harm))
                )
            else:
                e_harm_interp = energies[i, :]
                b_harm_interp = brilliance[i, :]

            e_next_harm = input_energies[i + 1, :]
            b_next_harm = input_brilliance[i + 1, :]
            idx = _np.argsort(e_next_harm)
            e_next_harm = e_next_harm[idx]
            b_next_harm = b_next_harm[idx]

            e_next_harm_interp = _np.linspace(
                _np.min(e_next_harm), _np.max(e_next_harm), 2001
            )
            b_next_harm_interp = _np.exp(
                _np.interp(
                    e_next_harm_interp, e_next_harm, _np.log(b_next_harm)
                )
            )

            max_e_harm = _np.nanmax(e_harm_interp)
            min_e_harm = _np.nanmin(e_harm_interp)
            max_e_next_harm = _np.nanmax(e_next_harm_interp)
            min_e_next_harm = _np.nanmin(e_next_harm_interp)

            flag_pre_processing = False
            if max_e_harm >= min_e_next_harm:
                flag_pre_processing = True

                min_abs = _np.max((min_e_harm, min_e_next_harm))
                max_abs = _np.min((max_e_harm, max_e_next_harm))
                energy_intersect = _np.linspace(min_abs, max_abs, 2001)

                b_harm_intersect = _np.interp(
                    energy_intersect, e_harm_interp, b_harm_interp
                )
                b_next_harm_intersect = _np.interp(
                    energy_intersect, e_next_harm_interp, b_next_harm_interp
                )

                idcs_bigger = _np.where(
                    b_next_harm_intersect >= b_harm_intersect
                )

                ecross = energy_intersect[_np.min(idcs_bigger)]

                idx_cut_e1 = _np.nanargmin(
                    _np.abs(e_harm_interp - ecross - superp_value)
                )
                idx_cut_e3 = _np.nanargmin(
                    _np.abs(e_next_harm_interp - ecross + superp_value)
                )

                e_harm = e_harm_interp[:idx_cut_e1]
                b_harm = b_harm_interp[:idx_cut_e1]

                e_next_harm = e_next_harm_interp[idx_cut_e3:]
                b_next_harm = b_next_harm_interp[idx_cut_e3:]

            e_harm_new = _np.copy(e_harm)
            b_harm_new = _np.copy(b_harm)
            e_harm_new.resize(2001)
            b_harm_new.resize(2001)
            idx = len(e_harm) - 1
            e_harm_new[idx:] = _np.full(len(e_harm_new[idx:]), _np.nan)
            b_harm_new[idx:] = _np.full(len(e_harm_new[idx:]), _np.nan)

            e_next_harm_new = _np.copy(e_next_harm)
            b_next_harm_new = _np.copy(b_next_harm)
            e_next_harm_new.resize(2001)
            b_next_harm_new.resize(2001)
            idx = len(e_next_harm) - 1
            e_next_harm_new[idx:] = _np.full(
                len(e_next_harm_new[idx:]), _np.nan
            )
            b_next_harm_new[idx:] = _np.full(
                len(e_next_harm_new[idx:]), _np.nan
            )

            energies[i, :] = e_harm_new
            brilliance[i, :] = b_harm_new

            energies[i + 1, :] = e_next_harm_new
            brilliance[i + 1, :] = b_next_harm_new

        return energies, brilliance


class SpectraInterface:
    """Spectra Interface class."""

    def __init__(self):
        """Class constructor."""
        self._accelerator = SIRIUS.StorageRing()
        self._calc = Calc(self._accelerator)
        self._sources = None
        self._energies = None
        self._brilliances = None
        self._fluxes = None
        self._target_energy = None
        self._flux_density_matrix = None
        self._info_matrix_flux_density = None
        self._flux_matrix = None
        self._info_matrix_flux = None
        self._brilliance_matrix = None
        self._info_matrix_brilliance = None
        self._flag_brill_processed = False
        self._flag_flux_processed = False

    @property
    def accelerator(self):
        """Accelerator parameters.

        Returns:
            StorageRingParameters object: class to config accelerator.
        """
        return self._accelerator

    @property
    def calc(self):
        """CalcFlux object.

        Returns:
            CalcFlux object: Class to calculate flux
        """
        return self._calc

    @property
    def sources(self):
        """Sources list.

        Returns:
            List: List of sources objects
        """
        return self._sources

    @property
    def energies(self):
        """List of energies for each undulator.

        Returns:
            list of numpy arrays: Energies for each undulator, for each
             harmonic.
        """
        return self._energies

    @property
    def brilliances(self):
        """List of brilliances for each undulator.

        Returns:
            list of numpy arrays: Brilliances for each undulator, for each
             harmonic.
        """
        return self._brilliances

    @property
    def fluxes(self):
        """List of fluxes for each undulator.

        Returns:
            list of numpy arrays: Fluxes for each undulator, for each
             harmonic.
        """
        return self._fluxes

    @property
    def target_energy(self):
        """Target energy.

        Returns:
            float: Target energy to analyse.
        """
        return self._target_energy

    @property
    def flux_density_matrix(self):
        """Flux density matrix.

        Returns:
            Array: Flux density matrix to analyse.
        """
        return self._flux_density_matrix

    @property
    def flux_matrix(self):
        """Flux matrix.

        Returns:
            Array: Flux matrix to analyse.
        """
        return self._flux_matrix

    @property
    def brilliance_matrix(self):
        """Brilliance matrix.

        Returns:
            Array: brilliance matrix to analyse.
        """
        return self._brilliance_matrix

    @property
    def info_matrix_flux_density(self):
        """Information about the respective undulators in
                the flux density matrix.

        Returns:
            Array: Undulators information to analyse.
        """
        return self._info_matrix_flux_density

    @property
    def info_matrix_flux(self):
        """Information about the respective undulators in the flux matrix.

        Returns:
            Array: Undulators information to analyse.
        """
        return self._info_matrix_flux

    @property
    def info_matrix_brilliance(self):
        """Information about the respective undulators in the brilliance matrix.

        Returns:
            Array: Undulators information to analyse.
        """
        return self._info_matrix_brilliance

    @accelerator.setter
    def accelerator(self, value):
        self._accelerator = value
        self._calc._accelerator = value

    @sources.setter
    def sources(self, value):
        self._sources = value

    @target_energy.setter
    def target_energy(self, value):
        self._target_energy = value

    @staticmethod
    def calc_rms(x, f_x):
        """RMS function.

        Args:
            x (numpy array): x values
            f_x (numpy array): f(x) values

        Returns:
            float: RMS Value
        """
        return _np.sqrt(
            _np.sum(f_x * _np.square(x)) / _np.sum(f_x)
            - (_np.sum(f_x * x) / _np.sum(f_x)) ** 2
        )

    @staticmethod
    def _truncate_at_intersections(x_list, y_list, superb=2e3):
        """Intersection function.

        Args:
            x_list (list): list of arrays.
            y_list (list): list of arrays.
            superb (float): extrapolation value of the intersection point.

        Returns:
            x_list_trunc (list): list of arrays.
            y_list_trunc (list): list of arrays.
        """
        n = len(x_list)
        xis = list()
        x_list_trunc = list()
        y_list_trunc = list()

        flag = True

        for i in range(n - 1):
            x1, y1 = x_list[i], y_list[i]
            x2, y2 = x_list[i + 1], y_list[i + 1]
            x_common = _np.linspace(
                max(x1.min(), x2.min()), min(x1.max(), x2.max()), 500
            )
            y1c = _np.interp(x_common, x1, y1)
            y2c = _np.interp(x_common, x2, y2)
            d = y1c - y2c
            idx = _np.where(d[:-1] * d[1:] < 0)[0]
            if len(idx) == 0:
                xis.append(x_list[i + 1].min())
                mask = x_list[i] < x_list[i + 1].min() + superb
                x_list_trunc.append(x_list[i][mask])
                y_list_trunc.append(y_list[i][mask])
                flag = False
                continue
            i0 = idx[0]
            xa, xb = x_common[i0], x_common[i0 + 1]
            da, db = d[i0], d[i0 + 1]
            t = -da / (db - da)
            xi = xa + t * (xb - xa)
            xis.append(xi)

        if flag:
            mask = x_list[0] < x_list[1].min() + superb
            x_list_trunc.append(x_list[0][mask])
            y_list_trunc.append(y_list[0][mask])

        for i in range(n - 1):
            xi_left = xis[i + 1 - 1] if i + 1 > 0 else -_np.inf
            xi_right = xis[i + 1] if i + 1 < n - 1 else _np.inf
            mask = (x_list[i + 1] > xi_left - superb) & (
                x_list[i + 1] < xi_right + superb
            )
            x_list_trunc.append(x_list[i + 1][mask])
            y_list_trunc.append(y_list[i + 1][mask])

        return x_list_trunc, y_list_trunc

    def export_data(data: dict, filename: str):
        """Export data function.

        Args:
            data (dict): data
            filename (str): file name

        Returns:
            None
        """
        with open('{:}.json'.format(filename), 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def apply_phase_error_matrix(self, values, harm, rec_param=True):
        """Add phase errors.

        Args:
            values (numpy 2d array): It can be brilliance of flux
            harm (int): Harmonic number
            rec_param (bool, optional): Use recovery params. Defaults to True.

        Returns:
            numpy 2d array: brilliance of flux with phase errors
        """
        fname = REPOS_PATH + '/files/phase_errors_fit.txt'
        data = _np.genfromtxt(fname, unpack=True, skip_header=1)
        h = data[:, 0]
        ph_err1 = data[:, 1]
        ph_err2 = data[:, 2]
        idx = _np.argmin(_np.abs(harm - h))
        if rec_param:
            ph = ph_err2[idx]
        else:
            ph = ph_err1[idx]
        return values * ph

    def _parallel_calc_brilliance_curve(  # noqa: C901
        self, args
    ):
        (
            source,
            accelerator,
            extraction_point,
            emax,
            harmonic_range,
            nr_pts_k,
            x_accep,
            kmin,
        ) = args

        # Spectra Parameters Copy
        spectra_calc = copy.deepcopy(self)
        spectra_calc.accelerator = accelerator

        if source.source_type != 'bendingmagnet':
            if source.min_gap == 0:
                kmax = source.calc_max_k(spectra_calc.accelerator)
            else:
                beff = source.get_beff(source.min_gap / source.period)
                kmax = source.undulator_b_to_k(b=beff, period=source.period)

            if source.source_type == 'wiggler':
                b_max = source.undulator_k_to_b(kmax, source.period)
                spectra_calc.calc.source_type = (
                    spectra_calc.calc.SourceType.wiggler
                )
                spectra_calc.calc.method = (
                    spectra_calc.calc.CalcConfigs.Method.far_field
                )
                spectra_calc.calc.indep_var = (
                    spectra_calc.calc.CalcConfigs.Variable.energy
                )
                spectra_calc.calc.output_type = (
                    spectra_calc.calc.CalcConfigs.Output.flux_density
                )
                spectra_calc.calc.period = source.period
                spectra_calc.calc.by_peak = b_max
                spectra_calc.calc.ky = kmax
                spectra_calc.calc.observation_angle = [0, 0]
                spectra_calc.calc.energy_range = [1, emax]
                spectra_calc.calc.energy_step = 50
            else:
                spectra_calc.calc.output_type = (
                    spectra_calc.calc.CalcConfigs.Output.brilliance
                )
                spectra_calc.calc._add_phase_errors = source.add_phase_errors
                spectra_calc.calc._use_recovery_params = (
                    source.use_recovery_params
                )
                spectra_calc.calc.indep_var = (
                    spectra_calc.calc.CalcConfigs.Variable.k
                )
                spectra_calc.calc.method = (
                    spectra_calc.calc.CalcConfigs.Method.wigner
                )
                spectra_calc.calc.slice_x = 0
                spectra_calc.calc.slice_px = 0
                spectra_calc.calc.slice_y = 0
                spectra_calc.calc.slice_py = 0
                spectra_calc.calc.harmonic_range = harmonic_range
                spectra_calc.calc.k_nr_pts = nr_pts_k

                polarization = source.polarization
                if polarization == 'hp':
                    spectra_calc.calc.source_type = (
                        spectra_calc.calc.SourceType.horizontal_undulator
                    )
                    spectra_calc.calc.by_peak = 1
                elif polarization == 'vp':
                    spectra_calc.calc.source_type = (
                        spectra_calc.calc.SourceType.vertical_undulator
                    )
                    spectra_calc.calc.bx_peak = 1
                elif polarization == 'cp':
                    spectra_calc.calc.source_type = (
                        spectra_calc.calc.SourceType.elliptic_undulator
                    )
                    spectra_calc.calc.bx_peak = 1
                    spectra_calc.calc.by_peak = source.fields_ratio
                else:
                    return

                spectra_calc.calc.k_range = [kmin, kmax]
                spectra_calc.calc.period = source.period

        else:
            b = source.b_peak
            spectra_calc.calc.source_type = (
                spectra_calc.calc.SourceType.bending_magnet
            )
            spectra_calc.calc.method = (
                spectra_calc.calc.CalcConfigs.Method.far_field
            )
            spectra_calc.calc.indep_var = (
                spectra_calc.calc.CalcConfigs.Variable.energy
            )
            spectra_calc.calc.slit_acceptance = x_accep
            spectra_calc.calc.output_type = (
                spectra_calc.calc.CalcConfigs.Output.flux_density
            )
            spectra_calc.calc.by_peak = b
            spectra_calc.calc.observation_angle = [0, 0]
            spectra_calc.calc.energy_range = [1, emax]
            spectra_calc.calc.energy_step = 50

        spectra_calc.calc.length = source.source_length

        if extraction_point is not None:
            if extraction_point in list(
                spectra_calc.accelerator.extraction_dict.keys()
            ):
                spectra_calc.accelerator.set_extraction_point(extraction_point)
            else:
                raise ValueError('Invalid extraction point.')

        spectra_calc.calc.set_config()
        spectra_calc.calc.run_calculation()

        energies = spectra_calc.calc.energies
        brilliances = spectra_calc.calc.brilliance
        del spectra_calc

        return energies, brilliances

    def calc_brilliance_curve(  # noqa: C901
        self,
        harmonic_range=(1, 5),
        nr_pts_k=15,
        kmin=0.2,
        emax=20e3,
        x_accep=1,
        extraction_points=None,
        export_data=False,
        filename='data_brilliance',
        superp_value=250,
        process_curves=True,
    ):
        """Calc brilliance curve.

        Args:
            harmonic_range (list, optional): List of desired harmonics.
             Defaults to [1, 5].
            nr_pts_k (int, optional): Number of k points. Defaults to 15.
            kmin (float): Minimum k value. Defaults to 0.2
            emax (float): Max value of energy for dipoles and wigglers.
            x_accep (float): X acceptance for bending magnet radiation.
            extraction_points (list of string): List of extraction points for each
             source.
            export_data (bool, optional): to export data.
             export_data. Defaults to False.
            filename (str, optional): filename.
             filename. Defaults to 'data'.
            superp_value (int, optional): Desired value of energy
             superposition. Defaults to 250.
            process_curves (bool, optional): If true energy superposition will
             be processed. Defaults to True.
        """
        self._flag_brill_processed = False
        self.calc._slit_shape = ''
        source_list = self.sources
        energies = list()
        brilliances = list()
        flag_bend = False

        if 'list' not in str(type(self.accelerator)):
            accelerators = list()
            for i, source in enumerate(source_list):
                accelerators.append(self.accelerator)
        else:
            accelerators = self.accelerator

        if 'list' not in str(type(harmonic_range)):
            harmonic_ranges = list()
            for i, source in enumerate(source_list):
                harmonic_ranges.append(harmonic_range)
        else:
            harmonic_ranges = harmonic_range

        arglist = []
        for i, source in enumerate(source_list):
            if (
                source.source_type == 'wiggler'
                or source.source_type == 'bendingmagnet'
            ):
                flag_bend = True
            arglist += [
                (
                    source,
                    accelerators[i],
                    extraction_points[i],
                    emax,
                    (1, 2)
                    if hasattr(source, 'polarization')
                    and source.polarization == 'cp'
                    else harmonic_ranges[i],
                    nr_pts_k,
                    x_accep,
                    kmin,
                )
            ]

        # Parallel calculations
        num_processes = multiprocessing.cpu_count()
        data = []
        with multiprocessing.Pool(processes=num_processes - 1) as parallel:
            data = parallel.map(self._parallel_calc_brilliance_curve, arglist)

        # Assembly data
        for i, result in enumerate(data):
            energies.append(result[0])
            brilliances.append(result[1])

        if flag_bend:
            energies = _np.array(energies, dtype=object)
            brilliances = _np.array(brilliances, dtype=object)
        else:
            energies = _np.array(energies)
            brilliances = _np.array(brilliances)

        self._energies = energies
        self._brilliances = brilliances

        if export_data:
            if process_curves is True:
                energies = list()
                brilliances = list()
                for i, source in enumerate(self.sources):
                    if (
                        source.source_type != 'wiggler'
                        and source.source_type != 'bendingmagnet'
                    ):
                        input_flux = self.brilliances[i][:, :]
                        input_energies = self.energies[i][:, :]
                        if input_energies.shape[0] > 1:
                            energies_, flux = (
                                self.calc.process_brilliance_curve(
                                    input_energies,
                                    input_flux,
                                    superp_value=superp_value,
                                )
                            )
                        else:
                            input_flux_b = input_flux[0]
                            input_energies_b = input_energies[0]
                            idx = _np.argsort(input_energies_b)
                            input_energies_b = input_energies_b[idx]
                            input_flux_b = input_flux_b[idx]
                            energies_ = _np.linspace(
                                _np.min(input_energies_b),
                                _np.max(input_energies_b),
                                2001,
                            )
                            flux = _np.interp(
                                energies_, input_energies_b, input_flux_b
                            )
                            energies_ = _np.reshape(
                                energies_, (1, _np.shape(energies_)[0])
                            )
                            flux = _np.reshape(flux, (1, _np.shape(flux)[0]))
                    else:
                        input_flux = _np.array(
                            self.brilliances[i], dtype=float
                        )
                        input_energies = _np.array(
                            self.energies[i], dtype=float
                        )
                        energies_ = _np.linspace(
                            _np.min(input_energies),
                            _np.max(input_energies),
                            2001,
                        )
                        flux = _np.interp(
                            energies_, input_energies, input_flux
                        )
                        energies_ = _np.reshape(
                            energies_, (1, _np.shape(energies_)[0])
                        )
                        flux = _np.reshape(flux, (1, _np.shape(flux)[0]))

                    energies.append(energies_)
                    brilliances.append(flux)
                energies = _np.array(energies, dtype=object)
                brilliances = _np.array(brilliances, dtype=object)

            data = dict()
            data['calc'] = 'Brilliances Curves'
            data['units'] = ['eV', 'ph/s/0.1%/mm²/mrad²/100mA']
            data['data'] = list()

            for i, source in enumerate(self.sources):
                data['data'].append(
                    {
                        'label': source.label,
                        'energies': energies[i].tolist(),
                        'brilliance': brilliances[i].tolist(),
                    }
                )

            self.export_data(data=data, filename='{:}'.format(filename))

    def _parallel_calc_flux_curve(self, args):
        (
            source,
            accelerator,
            extraction_point,
            energy_range,
            harmonic_range,
            nr_pts_k,
            slit_shape,
            slit_acceptance,
            kmin,
        ) = args

        # Spectra Parameters Copy
        spectra_calc = copy.deepcopy(self)
        spectra_calc.accelerator = accelerator

        if extraction_point is not None:
            if extraction_point in list(
                spectra_calc.accelerator.extraction_dict.keys()
            ):
                spectra_calc.accelerator.set_extraction_point(extraction_point)
            else:
                raise ValueError('Invalid extraction point.')

        if source.source_type != 'bendingmagnet':
            if source.min_gap != 0:
                beff = source.get_beff(source.min_gap / source.period)
                kmax = source.undulator_b_to_k(b=beff, period=source.period)
            else:
                kmax = source.calc_max_k(spectra_calc.accelerator)
            if source.source_type == 'wiggler':
                b_max = source.undulator_k_to_b(kmax, source.period)
                spectra_calc.calc.source_type = (
                    spectra_calc.calc.SourceType.wiggler
                )
                spectra_calc.calc.method = (
                    spectra_calc.calc.CalcConfigs.Method.far_field
                )
                spectra_calc.calc.indep_var = (
                    spectra_calc.calc.CalcConfigs.Variable.energy
                )
                spectra_calc.calc.output_type = (
                    spectra_calc.calc.CalcConfigs.Output.flux
                )
                spectra_calc.calc.slit_shape = slit_shape
                spectra_calc.calc.period = source.period
                spectra_calc.calc.by_peak = b_max
                spectra_calc.calc.ky = kmax
                spectra_calc.calc.observation_angle = [0, 0]
                spectra_calc.calc.slit_acceptance = slit_acceptance
                spectra_calc.calc.energy_range = energy_range
                spectra_calc.calc.energy_step = 1
            else:
                spectra_calc.calc._add_phase_errors = source.add_phase_errors
                spectra_calc.calc._use_recovery_params = (
                    source.use_recovery_params
                )
                spectra_calc.calc.output_type = (
                    spectra_calc.calc.CalcConfigs.Output.flux
                )
                spectra_calc.calc.indep_var = (
                    spectra_calc.calc.CalcConfigs.Variable.k
                )
                spectra_calc.calc.method = (
                    spectra_calc.calc.CalcConfigs.Method.far_field
                )
                spectra_calc.calc.slit_shape = slit_shape
                spectra_calc.calc.harmonic_range = harmonic_range
                spectra_calc.calc.k_nr_pts = nr_pts_k
                spectra_calc.calc.slit_acceptance = slit_acceptance

                polarization = source.polarization
                if polarization == 'hp':
                    spectra_calc.calc.source_type = (
                        spectra_calc.calc.SourceType.horizontal_undulator
                    )
                    spectra_calc.calc.by_peak = 1
                elif polarization == 'vp':
                    spectra_calc.calc.source_type = (
                        spectra_calc.calc.SourceType.vertical_undulator
                    )
                    spectra_calc.calc.bx_peak = 1
                elif polarization == 'cp':
                    spectra_calc.calc.source_type = (
                        spectra_calc.calc.SourceType.elliptic_undulator
                    )
                    spectra_calc.calc.bx_peak = 1
                    spectra_calc.calc.by_peak = source.fields_ratio
                else:
                    return
                spectra_calc.calc.k_range = [kmin, kmax]
                spectra_calc.calc.period = source.period

        else:
            b = source.b_peak
            spectra_calc.calc.source_type = (
                spectra_calc.calc.SourceType.bending_magnet
            )
            spectra_calc.calc.method = (
                spectra_calc.calc.CalcConfigs.Method.far_field
            )
            spectra_calc.calc.indep_var = (
                spectra_calc.calc.CalcConfigs.Variable.energy
            )
            spectra_calc.calc.output_type = (
                spectra_calc.calc.CalcConfigs.Output.flux
            )
            spectra_calc.calc.slit_shape = slit_shape
            spectra_calc.calc.observation_angle = [0, 0]
            spectra_calc.calc.slit_acceptance = slit_acceptance
            spectra_calc.calc.energy_range = energy_range
            spectra_calc.calc.energy_step = 1
            spectra_calc.calc.by_peak = b

        spectra_calc.calc.length = source.source_length

        spectra_calc.calc.set_config()
        spectra_calc.calc.run_calculation()

        energies = spectra_calc.calc.energies
        fluxes = spectra_calc.calc.flux
        del spectra_calc

        return energies, fluxes

    def calc_flux_curve(  # noqa: C901
        self,
        energy_range=[1, 5],
        harmonic_range=[1, 5],
        nr_pts_k=15,
        kmin=0.2,
        slit_shape='circslit',
        slit_acceptances=[[0, 0.04]],
        extraction_points=None,
        export_data=False,
        filename='data_flux',
        superp_value=250,
        process_curves=True,
    ):
        """Calc flux curves.

        Args:
            energy_range (list, optional): Energy range for wigglers and
             bending magnets. Defaults to [1, 5].
            harmonic_range (list, optional): List of desired harmonics.
             Defaults to [1, 5].
            nr_pts_k (int, optional): Number of k points. Defaults to 15.
            kmin (float): Minimum k value. Defaults to 0.2
            slit_shape (str, optional): Circular or rectangular.
             Defaults to "circslit".
            slit_acceptances (list, optional): Slit acceptance.
             Defaults to [0, 0.04].
            extraction_points (list of string): List of extraction points for each
             source.
            export_data (bool, optional): to export data.
             export_data. Defaults to False.
            filename (str, optional): filename.
             filename. Defaults to 'data'.
            superp_value (int, optional): Desired value of energy
             superposition. Defaults to 250.
            process_curves (bool, optional): If true energy superposition will
             be processed. Defaults to True.

        Raises:
            ValueError: _description_
        """
        self._flag_flux_processed = False
        source_list = self.sources
        energies = list()
        fluxes = list()
        slit_acceptances = _np.array(slit_acceptances)
        if slit_acceptances.shape[0] == 1:
            slit_acceptances = _np.full(
                (len(source_list), 2), slit_acceptances[0]
            )
        slit_acceptances = slit_acceptances.tolist()
        flag_bend = False

        if 'list' not in str(type(self.accelerator)):
            accelerators = list()
            for i, source in enumerate(source_list):
                accelerators.append(self.accelerator)
        else:
            accelerators = self.accelerator

        if 'list' not in str(type(harmonic_range)):
            harmonic_ranges = list()
            for i, source in enumerate(source_list):
                harmonic_ranges.append(harmonic_range)
        else:
            harmonic_ranges = harmonic_range

        arglist = []
        for i, source in enumerate(source_list):
            if (
                source.source_type == 'wiggler'
                or source.source_type == 'bendingmagnet'
            ):
                flag_bend = True
            arglist += [
                (
                    source,
                    accelerators[i],
                    extraction_points[i],
                    energy_range,
                    (1, 2)
                    if hasattr(source, 'polarization')
                    and source.polarization == 'cp'
                    else harmonic_ranges[i],
                    nr_pts_k,
                    slit_shape,
                    slit_acceptances[i],
                    kmin,
                )
            ]

        # Parallel calculations
        num_processes = multiprocessing.cpu_count()
        data = []
        with multiprocessing.Pool(processes=num_processes - 1) as parallel:
            data = parallel.map(self._parallel_calc_flux_curve, arglist)

        # Assembly data
        for i, result in enumerate(data):
            energies.append(result[0])
            fluxes.append(result[1])

        if flag_bend:
            energies = _np.array(energies, dtype=object)
            fluxes = _np.array(fluxes, dtype=object)
        else:
            energies = _np.array(energies)
            fluxes = _np.array(fluxes)

        self._energies = energies
        self._fluxes = fluxes

        if export_data:
            if process_curves is True:
                energies = list()
                fluxes = list()
                for i, source in enumerate(self.sources):
                    if (
                        source.source_type != 'wiggler'
                        and source.source_type != 'bendingmagnet'
                    ):
                        input_flux = self.fluxes[i][:, :]
                        input_energies = self.energies[i][:, :]
                        if input_energies.shape[0] > 1:
                            energies_, flux = (
                                self.calc.process_brilliance_curve(
                                    input_energies,
                                    input_flux,
                                    superp_value=superp_value,
                                )
                            )
                        else:
                            input_flux_b = input_flux[0]
                            input_energies_b = input_energies[0]
                            idx = _np.argsort(input_energies_b)
                            input_energies_b = input_energies_b[idx]
                            input_flux_b = input_flux_b[idx]
                            energies_ = _np.linspace(
                                _np.min(input_energies_b),
                                _np.max(input_energies_b),
                                2001,
                            )
                            flux = _np.interp(
                                energies_, input_energies_b, input_flux_b
                            )
                            energies_ = _np.reshape(
                                energies_, (1, _np.shape(energies_)[0])
                            )
                            flux = _np.reshape(flux, (1, _np.shape(flux)[0]))
                    else:
                        input_flux = _np.array(self.fluxes[i], dtype=float)
                        input_energies = _np.array(
                            self.energies[i], dtype=float
                        )
                        energies_ = _np.linspace(
                            _np.min(input_energies),
                            _np.max(input_energies),
                            2001,
                        )
                        flux = _np.interp(
                            energies_, input_energies, input_flux
                        )
                        energies_ = _np.reshape(
                            energies_, (1, _np.shape(energies_)[0])
                        )
                        flux = _np.reshape(flux, (1, _np.shape(flux)[0]))

                    energies.append(energies_)
                    fluxes.append(flux)
                energies = _np.array(energies, dtype=object)
                fluxes = _np.array(fluxes, dtype=object)

            data = dict()
            data['calc'] = 'Flux Curves'
            data['units'] = ['eV', 'ph/s/0.1%/100mA']
            data['data'] = list()

            for i, source in enumerate(self.sources):
                data['data'].append(
                    {
                        'label': source.label,
                        'energies': energies[i].tolist(),
                        'flux': fluxes[i].tolist(),
                    }
                )

            self.export_data(data=data, filename='{:}'.format(filename))

    def _parallel_calc_flux_fpmethod(self, args):
        (
            source,
            target_k,
            target_energy,
            slit_shape,
            slit_acceptance,
            observation_angle,
            distance_from_source,
            _,
            _,
            harmonic,
        ) = args

        spectra_calc: SpectraInterface = copy.deepcopy(self)
        spectra_calc.calc.source_type = source.source_type
        spectra_calc.calc.indep_var = (
            spectra_calc.calc.CalcConfigs.Variable.energy
        )
        spectra_calc.calc.method = (
            spectra_calc.calc.CalcConfigs.Method.fixedpoint_far_field
        )
        spectra_calc.calc.output_type = (
            spectra_calc.calc.CalcConfigs.Output.flux
        )

        spectra_calc.calc.slit_shape = slit_shape
        spectra_calc.calc.slit_acceptance = slit_acceptance
        spectra_calc.calc.observation_angle = observation_angle

        if source.source_type != 'bendingmagnet':
            source_polarization = source.polarization
            spectra_calc.calc.period = source.period
            spectra_calc.calc.length = source.source_length

            if source_polarization == 'hp':
                spectra_calc.calc.ky = target_k
            elif source_polarization == 'vp':
                spectra_calc.calc.kx = target_k
            else:
                spectra_calc.calc.kx = target_k / _np.sqrt(
                    1 + source.fields_ratio**2
                )
                spectra_calc.calc.ky = (
                    spectra_calc.calc.kx * source.fields_ratio
                )
        else:
            spectra_calc.calc.by = source.b_peak

        spectra_calc.calc.distance_from_source = distance_from_source
        spectra_calc.calc.target_energy = target_energy

        spectra_calc.calc.set_config()
        spectra_calc.calc.run_calculation()
        flux_total = spectra_calc.calc.flux
        if source.source_type != 'bendingmagnet':
            if source.use_recovery_params and source.add_phase_errors:
                flux_total = spectra_calc.apply_phase_error_matrix(
                    values=flux_total,
                    harm=harmonic,
                    rec_param=spectra_calc.use_recovery_params,
                )
        return flux_total[0]

    def calc_flux_curve_generic(
        self,
        und,
        emax=20e3,
        slit_shape='retslit',
        slit_acceptance=(0.060, 0.060),
        observation_angle=(0, 0),
        distance_from_source=30,
        k_nr_pts=1,
        deltak=0.99,
        even_harmonic=False,
        superb=1e3,
        kmin=0.1,
    ):
        """Calculate flux curve generic, at res, out res, even and odd harmonic.

        Args:
            und (Undulator object): Must be an object from undulator class.
            emax (float): Máx energy range to calculate [eV]
            slit_shape (str, optional): Slit shape "retslit" or "circslit".
                Defaults to "retslit".
            slit_acceptance (tuple, optional): Slit acceptances.
                Defaults to (0.060, 0.060).
            observation_angle (tuple, optional): Slit position.
                Defaults to (0.060, 0.060).
            kmin (float): Minimum K allowed.
            distance_from_source (float, optional): Distance from source.
                Defaults to 23.
            method (str, optional): Method of calc. Defaults to "farfield".
            k_nr_pts (int, optional): Number of K points around
                                        of ressonance k.
                Defaults to 1 to use ressonance k.
            dk (float, optional): Rate for change of k
            even_harmonic (bool, optional): If it is false it will be
                                            calculated for the even harmonic
            superb (int, optional): Extrapolation of the intersection
                                   of the curve

        Returns:
            tuple: Fluxes, and Energies.
        """
        if und.min_gap == 0:
            source_k_max = und.calc_max_k(self.accelerator)
        else:
            beff = und.get_beff(und.min_gap / und.period)
            source_k_max = und.undulator_b_to_k(b=beff, period=und.period)

        first_hamonic_energy = und.get_harmonic_energy(
            1, self.accelerator.gamma, 0, und.period, source_k_max
        )

        n = int(emax / first_hamonic_energy)
        if n > 0:
            n_harmonic = n - 1 if n % 2 == 0 else n
        else:
            n_harmonic = 1
        ns = _np.linspace(1, n_harmonic, n_harmonic)
        if not even_harmonic:
            ns = ns[::2]
        else:
            ns = ns[1::2]
        ks = _np.linspace(source_k_max, kmin, 41)

        arglist = []
        for i, harmonic in enumerate(ns):
            for j, k in enumerate(ks):
                e = und.get_harmonic_energy(
                    harmonic, self.accelerator.gamma, 0, und.period, k
                )
                if (
                    e < (harmonic + 2) * first_hamonic_energy + 5e3
                    and e < emax + 2e3
                ):
                    dks = _np.linspace(k, k * deltak, k_nr_pts)
                    for w, dk in enumerate(dks):
                        arglist += [
                            (
                                und,
                                dk,
                                e,
                                slit_shape,
                                slit_acceptance,
                                observation_angle,
                                distance_from_source,
                                j,
                                w,
                                harmonic,
                            )
                        ]

        data = []
        num_process = multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=num_process - 1) as parallel:
            data = parallel.map(self._parallel_calc_flux_fpmethod, arglist)

        arglist = _np.array(arglist, dtype='object')
        arglist = arglist[:, [1, 2, 7, 8]]
        result = _np.array(data)

        if k_nr_pts > 1:
            idx_broke = list(
                _np.where(
                    (arglist[:-1, 2] != arglist[1:, 2])
                    | (arglist[:-1, 3] == arglist[1:, 3])
                )[0]
            )
            idx_broke.append(len(arglist) - 1)

            i_start = 0
            filter_arglist = []
            filter_result = []

            for i in idx_broke:
                collection_arg = []
                collection_result = []
                for j in range(i_start, i + 1):
                    collection_arg.append(list(arglist[j]))
                    collection_result.append(result[j])
                i_start = i + 1
                filter_arglist.append(collection_arg)
                filter_result.append(collection_result)

            filter_arglist = _np.array(filter_arglist)
            filter_result = _np.array(filter_result)

            best_result = []
            best_arglist = []

            for i, flux_values in enumerate(filter_result):
                if k_nr_pts > 1:
                    fs_result = _np.flip(filter_result[i, :])
                    ks_result = _np.flip(filter_arglist[i, :, 0])
                    es_result = _np.flip(filter_arglist[i, :, 1])
                    hs_result = _np.flip(filter_arglist[i, :, 2])

                    spl = make_interp_spline(ks_result, fs_result, k=3)
                    smooth_ks = _np.linspace(
                        ks_result.min(), ks_result.max(), 300
                    )
                    smooth_fs = spl(smooth_ks)

                    best_result.append(smooth_fs[_np.argmax(smooth_fs)])
                    best_arglist.append(
                        [
                            smooth_ks[_np.argmax(smooth_fs)],
                            es_result[0],
                            hs_result[0],
                        ]
                    )
                else:
                    best_result.append(flux_values[0])
                    best_arglist.append(filter_arglist[i][0])

            best_arglist = _np.array(best_arglist)
            best_result = _np.array(best_result)
        else:
            best_arglist = arglist
            best_result = result

        idx_broke = list(_np.where(best_arglist[:, 2] == 0)[0][1:] - 1)
        idx_broke.append(len(best_arglist) - 1)

        i_start = 0
        harmonic_arglist = []
        harmonic_result = []

        for i in idx_broke:
            collection_arg = []
            collection_result = []
            for j in range(i_start, i + 1):
                collection_arg.append(list(best_arglist[j]))
                collection_result.append(best_result[j])
            i_start = i + 1
            harmonic_arglist.append(_np.array(collection_arg))
            harmonic_result.append(_np.array(collection_result))

        fs = harmonic_result
        es = [harmonic[:, 1] for harmonic in harmonic_arglist]

        es, fs = self._truncate_at_intersections(
            x_list=es, y_list=fs, superb=superb
        )

        return fs, es

    def _calc_flux_density(
        self,
        target_energy: float,
        source_period: float,
        source_length: float,
        target_k: float,
        distance_from_source: float,
    ):
        """Calculate flux density for one k value.

        Args:
            target_energy (float): target energy of radiation [eV].
            source_period (float): undulator period [mm].
            source_length (float): undulator length [m].
            target_k (float): K value.
            distance_from_source (float): Distance from source.

        Returns:
            float: Flux density value
        """
        self._target_energy = target_energy
        und: Undulator = self._und

        # Spectra Initialization
        spectra = copy.deepcopy(self)
        spectra.accelerator.set_extraction_point(
            self.accelerator.extraction_point
        )

        # Spectra Configuration
        spectra.accelerator.zero_emittance = self.accelerator.zero_emittance
        spectra.accelerator.zero_energy_spread = (
            self.accelerator.zero_emittance
        )

        if und.polarization == 'hp':
            spectra.calc.source_type = (
                spectra.calc.SourceType.horizontal_undulator
            )
            spectra.calc.ky = target_k
        elif und.polarization == 'vp':
            spectra.calc.source_type = (
                spectra.calc.SourceType.vertical_undulator
            )
            spectra.calc.kx = target_k
        else:
            spectra.calc.source_type = (
                spectra.calc.SourceType.elliptic_undulator
            )
            spectra.calc.kx = target_k / _np.sqrt(1 + und.fields_ratio**2)
            spectra.calc.ky = spectra.calc.kx * und.fields_ratio

        spectra.calc.output_type = spectra.calc.CalcConfigs.Output.flux_density
        spectra.calc.method = spectra.calc.CalcConfigs.Method.far_field
        spectra.calc.output_type = self.calc.output_type

        spectra.calc.distance_from_source = distance_from_source
        spectra.calc.observation_angle = [0, 0]
        spectra.calc.energy_range = [
            self._target_energy,
            self._target_energy + 0.01,
        ]
        spectra.calc.energy_step = 0.01

        # Spectra calculation
        spectra.calc.period = source_period
        spectra.calc.length = source_length
        spectra.calc.set_config()
        spectra.calc.run_calculation()

        result = _np.max(spectra.calc.flux)
        del spectra

        return [result, target_k]

    def _parallel_calc_flux_density(self, args):
        target_k, period, length, _, distance = args
        return self._calc_flux_density(
            self._target_energy, period, length, target_k, distance
        )

    def calc_flux_density_matrix(  # noqa: C901
        self,
        target_energy: float,
        und: Undulator,
        periods,
        lengths,
        harmonics,
        kmin,
        distance_from_source=23,
    ):
        """Calc flux density matrix.

        Args:
            target_energy (float): Target energy to evaluate matrix [eV]
            und (Undulator object): Must be an object from undulator class.
            periods (1D Numpy array): Periods to evaluate calculation.
            lengths (1D Numpy array): Lengths to evaluate calculation.
            harmonics (1D numpy array): Harmonics - must be an array with ints.
            kmin (float): Minimum K allowed.
            distance_from_source (float, optional): Distance from source.
                Defaults to 23.

        Returns:
            tuple: Flux density matrix, and information matrix
                [k, period, length, n_harm].
        """
        n = harmonics
        gamma = self.accelerator.gamma
        self._target_energy = target_energy
        self._und = und

        # Arglist assembly
        arglist = []
        for i, length in enumerate(lengths):
            for j, period in enumerate(periods):
                self._und.period = period
                self._und.source_length = length
                k_max = self._und.calc_max_k(self.accelerator)
                ks = self._und.calc_k_target(
                    gamma, n, period, self._target_energy
                )
                isnan = _np.isnan(ks)
                idcs_nan = _np.argwhere(~isnan)
                idcs_max = _np.argwhere(ks < k_max)
                idcs_kmin = _np.argwhere(ks > kmin)
                idcs = _np.intersect1d(
                    idcs_nan.ravel(),
                    _np.intersect1d(idcs_max.ravel(), idcs_kmin.ravel()),
                )
                kres = ks[idcs]
                harm = n[idcs]
                if idcs.size == 0:
                    arglist += [
                        (
                            0,
                            period,
                            length,
                            1,
                            distance_from_source,
                        )
                    ]

                else:
                    for z, k in enumerate(kres):
                        arglist += [
                            (
                                k,
                                period,
                                length,
                                harm[z],
                                distance_from_source,
                            )
                        ]

        # Parallel calculations
        num_processes = multiprocessing.cpu_count()
        data = []
        with multiprocessing.Pool(processes=num_processes - 1) as parallel:
            data = parallel.map(self._parallel_calc_flux_density, arglist)

        arglist = _np.array(arglist, dtype='object')
        arglist = arglist[:, [0, 1, 2, 3]]
        result = _np.array(data)

        # Identification of breaks with equal length and equal periods
        idx_broke = list(
            _np.where(
                (arglist[:-1, 1] != arglist[1:, 1])
                | (arglist[:-1, 2] != arglist[1:, 2])
            )[0]
        )
        idx_broke.append(len(arglist) - 1)

        i_start = 0
        filter_arglist = []
        filter_result = []

        for i in idx_broke:
            collection_arg = []
            collection_result = []
            for j in range(i_start, i + 1):
                collection_arg.append(list(arglist[j]))
                collection_result.append(list(result[j]))
            i_start = i + 1
            filter_arglist.append(collection_arg)
            filter_result.append(collection_result)

        # Selection of the best results for a given period and length
        best_result = []
        info_unds = []

        for i, fluxs_densenties in enumerate(filter_result):
            arr = _np.array(fluxs_densenties)[:, 0]
            best_result.append(fluxs_densenties[_np.argmax(arr)])
            info_unds.append(filter_arglist[i][_np.argmax(arr)])

        best_result = _np.array(best_result)
        info_unds = _np.array(info_unds)

        # Flux Density Matrix Reassembly
        flux_density_matrix = best_result[:, 0]
        flux_density_matrix = flux_density_matrix.reshape(
            len(periods), len(lengths), order='F'
        )
        flux_density_matrix = flux_density_matrix.transpose()

        self._flux_density_matrix = flux_density_matrix
        self._info_matrix_flux_density = info_unds

        return flux_density_matrix, info_unds, und

    def _calc_flux(
        self,
        target_energy: float,
        source_period: float,
        source_length: float,
        target_k: float,
        slit_shape: str,
        slit_acceptance: list,
        distance_from_source: float,
        method: str,
        n_harmonic: int,
    ):
        """Calculate flux for one k value.

        Args:
            target_energy (float): target energy of radiation [eV].
            source_period (float): undulator period [mm].
            source_length (float): undulator length [m].
            target_k (float): K value.
            slit_shape (str): shape of slit acceptance 'retslit' or 'circslit'.
            slit_acceptance (list): slit aceeptance [mrad, mrad].
            distance_from_source (float): distance from the source [m]
            method (int): method to use in fixed point calculation 'farfield' or 'nearfield'
            n_harmonic (int): harmonic number to used in the calculation

        Returns:
            float: Flux value
        """  # noqa: E501
        if target_k == 0:
            return 0
        self._target_energy = target_energy
        und: Undulator = self._und

        # Spectra Initialization
        spectra = copy.deepcopy(self)
        spectra.accelerator.set_extraction_point(
            self.accelerator.extraction_point
        )

        # Spectra Configuration
        spectra.accelerator.zero_emittance = self.accelerator.zero_emittance
        spectra.accelerator.zero_energy_spread = (
            self.accelerator.zero_emittance
        )

        if und.polarization == 'hp':
            spectra.calc.source_type = (
                spectra.calc.SourceType.horizontal_undulator
            )
            spectra.calc.ky = target_k
        elif und.polarization == 'vp':
            spectra.calc.source_type = (
                spectra.calc.SourceType.vertical_undulator
            )
            spectra.calc.kx = target_k
        else:
            spectra.calc.source_type = (
                spectra.calc.SourceType.elliptic_undulator
            )
            spectra.calc.kx = target_k / _np.sqrt(1 + und.fields_ratio**2)
            spectra.calc.ky = spectra.calc.kx * und.fields_ratio

        if method == 'farfield':
            spectra.calc.method = (
                spectra.calc.CalcConfigs.Method.fixedpoint_far_field
            )
        elif method == 'nearfield':
            spectra.calc.method = (
                spectra.calc.CalcConfigs.Method.fixedpoint_near_field
            )

        spectra.calc.indep_var = spectra.calc.CalcConfigs.Variable.energy
        spectra.calc.output_type = spectra.calc.CalcConfigs.Output.flux

        if slit_shape == 'retslit':
            spectra.calc.slit_shape = (
                spectra.calc.CalcConfigs.SlitShape.rectangular
            )
        elif slit_shape == 'circslit':
            spectra.calc.slit_shape = (
                spectra.calc.CalcConfigs.SlitShape.circular
            )

        spectra.calc.target_energy = self._target_energy
        spectra.calc.distance_from_source = distance_from_source
        spectra.calc.observation_angle = [0, 0]
        spectra.calc.slit_acceptance = slit_acceptance

        # Spectra calculation
        spectra.calc.period = source_period
        spectra.calc.length = source_length
        spectra.calc.set_config()
        spectra.calc.run_calculation()
        if und.add_phase_errors:
            rec_param = und.use_recovery_params
            flux = spectra.apply_phase_error_matrix(
                _np.max(spectra.calc.flux), n_harmonic, rec_param=rec_param
            )
        else:
            flux = _np.max(spectra.calc.flux)

        del spectra

        return flux

    def _parallel_calc_flux(self, args):
        (
            target_k,
            period,
            length,
            n_harmonic,
            distance_from_source,
            slit_x,
            slit_y,
            method,
            slit_shape,
        ) = args
        slit_acceptance = [slit_x, slit_y]
        return self._calc_flux(
            self._target_energy,
            period,
            length,
            target_k,
            slit_shape,
            slit_acceptance,
            distance_from_source,
            method,
            n_harmonic,
        )

    def calc_flux_matrix(  # noqa: C901
        self,
        target_energy,
        und: Undulator,
        periods,
        lengths,
        harmonics,
        kmin,
        slit_shape='retslit',
        slit_acceptance=(0.230, 0.230),
        distance_from_source=23,
        method='farfield',
        nr_pts_k=1,
        k_range=0.99,
    ):
        """Calculate flux matrix.

        Args:
            target_energy (float): Target energy to evaluate matrix [eV]
            und (Undulator object): Must be an object from undulator class.
            periods (1D Numpy array): Periods to evaluate calculation.
            lengths (1D Numpy array): Lengths to evaluate calculation.
            harmonics (1D numpy array): Harmonics - must be an array with ints.
            kmin (float): Minimum K allowed.
            slit_shape (str, optional): Slit shape. Defaults to "retslit".
            slit_acceptance (tuple, optional): Slit acceptances.
                Defaults to (0.230, 0.230).
            distance_from_source (float, optional): Distance from source.
                Defaults to 23.
            method (str, optional): Method of calc. Defaults to "farfield".
            nr_pts_k (int, optional): Number of K points. Defaults to 1.
            k_range (float, optional): K range to evaluate detuning.
                Defaults to 0.99.

        Returns:
            tuple: Flux matrix, and information matrix
                [k, period, length, n_harm].
        """
        n = harmonics
        gamma = self.accelerator.gamma
        self._target_energy = target_energy
        self._und = und

        # Arglist assembly
        arglist = []
        for i, length in enumerate(lengths):
            for j, period in enumerate(periods):
                self._und.period = period
                self._und.source_length = length
                k_max = self._und.calc_max_k(self.accelerator)
                ks = self._und.calc_k_target(
                    gamma, n, period, self._target_energy
                )
                isnan = _np.isnan(ks)
                idcs_nan = _np.argwhere(~isnan)
                idcs_max = _np.argwhere(ks < k_max)
                idcs = _np.intersect1d(idcs_nan.ravel(), idcs_max.ravel())
                kres = ks[idcs]
                harm = n[idcs]
                if idcs.size == 0 or k_max < kmin:
                    arglist += [
                        (
                            0,
                            period,
                            length,
                            1,
                            distance_from_source,
                            slit_acceptance[0],
                            slit_acceptance[1],
                            method,
                            slit_shape,
                        )
                    ]

                else:
                    for z, k in enumerate(kres):
                        ks_probe = _np.linspace(k, k * k_range, nr_pts_k)
                        for k_probe in ks_probe:
                            arglist += [
                                (
                                    k_probe,
                                    period,
                                    length,
                                    harm[z],
                                    distance_from_source,
                                    slit_acceptance[0],
                                    slit_acceptance[1],
                                    method,
                                    slit_shape,
                                )
                            ]

        # Parallel calculations
        num_processes = multiprocessing.cpu_count()
        data = []
        with multiprocessing.Pool(processes=num_processes - 1) as parallel:
            data = parallel.map(self._parallel_calc_flux, arglist)

        # Data processing
        arglist = _np.array(arglist, dtype='object')
        arglist = arglist[:, [0, 1, 2, 3]]
        result = _np.array(data)

        # Identification of breaks with equal length and equal periods
        idx_broke = list(
            _np.where(
                (arglist[:-1, 1] != arglist[1:, 1])
                | (arglist[:-1, 2] != arglist[1:, 2])
            )[0]
        )
        idx_broke.append(len(arglist) - 1)

        i_start = 0
        filter_arglist = []
        filter_result = []

        for i in idx_broke:
            collection_arg = []
            collection_result = []
            for j in range(i_start, i + 1):
                collection_arg.append(list(arglist[j]))
                collection_result.append(result[j])
            i_start = i + 1
            filter_arglist.append(collection_arg)
            filter_result.append(collection_result)

        # Selection of the best results for a given period and length
        best_result = []
        info_unds = []

        for i, fluxs in enumerate(filter_result):
            arr = _np.array(fluxs)
            best_result.append(fluxs[_np.argmax(arr)])
            info_unds.append(filter_arglist[i][_np.argmax(arr)])

        best_result = _np.array(best_result)
        info_unds = _np.array(info_unds)

        # Flux Matrix Reassembly
        flux_matrix = best_result
        flux_matrix = flux_matrix.reshape(
            len(periods), len(lengths), order='F'
        )
        flux_matrix = flux_matrix.transpose()

        self._flux_matrix = flux_matrix
        self._info_matrix_flux = info_unds

        return flux_matrix, info_unds, und

    def _calc_brilliance(
        self,
        target_harmonic: float,
        source_period: float,
        source_length: float,
        target_k: float,
    ):
        """Calculate brilliance for one k value.

        Args:
            target_harmonic (float): target harmonic energy [eV].
            source_period (float): undulator period [mm].
            source_length (float): undulator length [m].
            target_k (float): K value.

        Returns:
            float: Brilliance value
        """
        if target_k == 0:
            return 0
        und: Undulator = self._und

        # Spectra Initialization
        spectra = copy.deepcopy(self)
        spectra.accelerator.set_extraction_point(
            self.accelerator.extraction_point
        )

        # Spectra Configuration
        spectra.accelerator.zero_emittance = self.accelerator.zero_emittance
        spectra.accelerator.zero_energy_spread = (
            self.accelerator.zero_emittance
        )

        spectra.calc.output_type = spectra.calc.CalcConfigs.Output.brilliance
        spectra.calc.method = spectra.calc.CalcConfigs.Method.fixedpoint_wigner
        spectra.calc.indep_var = spectra.calc.CalcConfigs.Variable.energy
        if und.polarization == 'hp':
            spectra.calc.source_type = (
                spectra.calc.SourceType.horizontal_undulator
            )
            spectra.calc.ky = target_k
        elif und.polarization == 'vp':
            spectra.calc.source_type = (
                spectra.calc.SourceType.vertical_undulator
            )
            spectra.calc.kx = target_k
        else:
            spectra.calc.source_type = (
                spectra.calc.SourceType.elliptic_undulator
            )
            spectra.calc.kx = target_k / _np.sqrt(1 + und.fields_ratio**2)
            spectra.calc.ky = spectra.calc.kx * und.fields_ratio

        spectra.calc.target_harmonic = int(target_harmonic)

        spectra.calc.slice_x = 0
        spectra.calc.slice_px = 0
        spectra.calc.slice_y = 0
        spectra.calc.slice_py = 0

        # Spectra calculation
        spectra.calc.period = source_period
        spectra.calc.length = source_length

        spectra.calc.set_config()
        spectra.calc.run_calculation()
        if und.add_phase_errors:
            rec_param = und.use_recovery_params
            brilliance = spectra.apply_phase_error_matrix(
                _np.max(spectra.calc.brilliance),
                target_harmonic,
                rec_param=rec_param,
            )
        else:
            brilliance = _np.max(spectra.calc.brilliance)

        del spectra

        return brilliance

    def _parallel_calc_brilliance(self, args):
        (
            target_k,
            period,
            length,
            n_harmonic,
        ) = args
        return self._calc_brilliance(n_harmonic, period, length, target_k)

    def calc_brilliance_matrix(  # noqa: C901
        self,
        target_energy: float,
        und: Undulator,
        periods,
        lengths,
        harmonics,
        kmin,
    ):
        """Calculate brilliance matrix.

        Args:
            target_energy (float): Target energy to evaluate matrix [eV]
            und (Undulator object): Must be an object from undulator class.
            periods (1D Numpy array): Periods to evaluate calculation.
            lengths (1D Numpy array): Lengths to evaluate calculation.
            harmonics (1D numpy array): Harmonics - must be an array with ints.
            kmin (float): Minimum K allowed.

        Returns:
            tuple: Brilliance matrix, and information matrix
                [k, period, length, n_harm].
        """
        n = harmonics
        gamma = self.accelerator.gamma
        self._target_energy = target_energy
        self._und = und

        # Arglist assembly
        arglist = []
        for i, length in enumerate(lengths):
            for j, period in enumerate(periods):
                self._und.period = period
                self._und.source_length = length
                k_max = self._und.calc_max_k(self.accelerator)
                ks = self._und.calc_k_target(
                    gamma, n, period, self._target_energy
                )
                isnan = _np.isnan(ks)
                idcs_nan = _np.argwhere(~isnan)
                idcs_max = _np.argwhere(ks < k_max)
                idcs = _np.intersect1d(idcs_nan.ravel(), idcs_max.ravel())
                kres = ks[idcs]
                harm = n[idcs]
                if idcs.size == 0 or k_max < kmin:
                    arglist += [
                        (
                            0,
                            period,
                            length,
                            1,
                        )
                    ]

                else:
                    for z, k in enumerate(kres):
                        arglist += [
                            (
                                k,
                                period,
                                length,
                                harm[z],
                            )
                        ]

        # Parallel calculations
        num_processes = multiprocessing.cpu_count()
        data = []
        with multiprocessing.Pool(processes=num_processes - 1) as parallel:
            data = parallel.map(self._parallel_calc_brilliance, arglist)

        arglist = _np.array(arglist, dtype='object')
        arglist = arglist[:, [0, 1, 2, 3]]
        result = _np.array(data)

        # Identification of breaks with equal length and equal periods
        idx_broke = list(
            _np.where(
                (arglist[:-1, 1] != arglist[1:, 1])
                | (arglist[:-1, 2] != arglist[1:, 2])
            )[0]
        )
        idx_broke.append(len(arglist) - 1)

        i_start = 0
        filter_arglist = []
        filter_result = []

        for i in idx_broke:
            collection_arg = []
            collection_result = []
            for j in range(i_start, i + 1):
                collection_arg.append(list(arglist[j]))
                collection_result.append(result[j])
            i_start = i + 1
            filter_arglist.append(collection_arg)
            filter_result.append(collection_result)

        # Selection of the best results for a given period and length
        best_result = []
        info_unds = []

        for i, brilliances in enumerate(filter_result):
            arr = _np.array(brilliances)
            best_result.append(brilliances[_np.argmax(arr)])
            info_unds.append(filter_arglist[i][_np.argmax(arr)])

        best_result = _np.array(best_result)
        info_unds = _np.array(info_unds)

        # Brilliance Matrix Reassembly
        brilliance_matrix = best_result
        brilliance_matrix = brilliance_matrix.reshape(
            len(periods), len(lengths), order='F'
        )
        brilliance_matrix = brilliance_matrix.transpose()

        self._brilliance_matrix = brilliance_matrix
        self._info_matrix_brilliance = info_unds

        return brilliance_matrix, info_unds, und

    def _calc_partial_power(
        self,
        source_period: float,
        source_length: float,
        target_k: float,
        slit_acceptance: list,
        distance_from_source: float,
        calcfarfield: int,
    ):
        """Calculate partial power for one k value.

        Args:
            target_energy (float): target energy of radiation [eV].
            source_period (float): undulator period [mm].
            source_length (float): undulator length [m].
            target_k (float): K value.
            slit_acceptance (list): slit aceeptance [mrad, mrad].
            distance_from_source (float): distance from the source [m]
            calcfarfield (int): method to use in fixed point calculation
                'farfield' 1 or 'nearfield' 0

        Returns:
            _type_: _description_
        """
        und: Undulator = self._und

        # Spectra Initialization
        spectra = copy.deepcopy(self)
        spectra.accelerator.set_extraction_point(
            self.accelerator.extraction_point
        )

        # Spectra Configuration
        spectra.accelerator.zero_emittance = self.accelerator.zero_emittance
        spectra.accelerator.zero_energy_spread = (
            self.accelerator.zero_emittance
        )

        if und.polarization == 'hp':
            spectra.calc.source_type = (
                spectra.calc.SourceType.horizontal_undulator
            )
            spectra.calc.ky = target_k
        elif und.polarization == 'vp':
            spectra.calc.source_type = (
                spectra.calc.SourceType.vertical_undulator
            )
            spectra.calc.kx = target_k
        else:
            spectra.calc.source_type = (
                spectra.calc.SourceType.elliptic_undulator
            )
            spectra.calc.kx = target_k / _np.sqrt(1 + und.fields_ratio**2)
            spectra.calc.ky = spectra.calc.kx * und.fields_ratio

        if calcfarfield == 1:
            spectra.calc.method = (
                spectra.calc.CalcConfigs.Method.fixedpoint_far_field
            )
        elif calcfarfield == 0:
            spectra.calc.method = (
                spectra.calc.CalcConfigs.Method.fixedpoint_near_field
            )

        spectra.calc.indep_var = spectra.calc.CalcConfigs.Variable.energy
        spectra.calc.output_type = spectra.calc.CalcConfigs.Output.power
        spectra.calc.slit_shape = (
            spectra.calc.CalcConfigs.SlitShape.rectangular
        )

        spectra.calc.target_energy = self._target_energy
        spectra.calc.distance_from_source = distance_from_source
        spectra.calc.observation_angle = [0, 0]
        spectra.calc.slit_acceptance = slit_acceptance

        # Spectra calculation
        spectra.calc.period = source_period
        spectra.calc.length = source_length
        spectra.calc.set_config()
        spectra.calc.run_calculation()

        return spectra.calc.power

    def _parallel_calc_partial_power(self, args):
        (
            target_k,
            period,
            length,
            distance_from_source,
            slit_x,
            slit_y,
            method,
        ) = args
        slit_acceptance = [slit_x, slit_y]
        return self._calc_partial_power(
            period,
            length,
            target_k,
            slit_acceptance,
            distance_from_source,
            method,
        )

    def calc_partial_power_from_matrix(
        self,
        data: tuple,
        slit_acceptance: list = [0.230, 0.230],  # noqa: B006
        distance_from_source: float = 30,
        method: str = 'farfield',
    ):
        """Calc partial power from matrix.

        Args:
            slit_acceptance (list): Slit acceptance [mrad, mrad].
             Defaults to [0.230, 0.230]
            distance_from_source (float): Distance from the source [m]
             Defaults to 10
            method (str): method to use in fixed point calculation 'farfield' or 'nearfield'
             Defaults to 'farfield'
            data (tuple): data especified to use in calculation
             First position 'flux matrix' or 'flux density matrix' or 'brilliance matrix'
             Second position unds matrix
        Returns:
            numpy array: partial power matrix.
        """
        if data is None:
            raise ValueError("'data' parameter has to be defined")

        unds_matrix = data[1]
        info_unds_matrix = unds_matrix

        calcfarfield = 1 if method == 'farfield' else 0

        # Arglist assembly
        arglist = info_unds_matrix[:, [0, 1, 2]]

        # Add distance from the source
        arglist = _np.c_[
            arglist, _np.ones((arglist.shape[0], 1)) * distance_from_source
        ]
        # Add slit x
        arglist = _np.c_[
            arglist, _np.ones((arglist.shape[0], 1)) * slit_acceptance[0]
        ]
        # Add slit y
        arglist = _np.c_[
            arglist, _np.ones((arglist.shape[0], 1)) * slit_acceptance[1]
        ]
        # Add method farfield or nearfield
        arglist = _np.c_[
            arglist, _np.ones((arglist.shape[0], 1)) * calcfarfield
        ]

        arglist = list(arglist)

        # Parallel calculations
        num_processes = multiprocessing.cpu_count()
        data = []
        with multiprocessing.Pool(processes=num_processes - 1) as parallel:
            data = parallel.map(self._parallel_calc_partial_power, arglist)

        arglist = _np.array(arglist)
        result = _np.array(data)

        # Partial power Matrix Reassembly
        pts_period = len(_np.where(arglist[:, 2] == arglist[0, 2])[0])
        pts_length = len(_np.where(arglist[:, 1] == arglist[0, 1])[0])

        partial_power_matrix = result.reshape(pts_length, pts_period)

        return partial_power_matrix

    def calc_proj_brilliance_with_phasespace(
        self,
        source,
        target_k,
        n_harmonic,
        r_range=[-0.02, 0.02],
        r_pts=101,
        rp_range=[-0.02, 0.02],
        rp_pts=101,
        direction='vertical',
    ):
        """Phase Space of Light Beam.

        Args:
            source: source light.
            target_k (float): target K.
            n_harmonic (int): harmonic number.
            r_range (list): size range to calculate.
            r_pts (int): points number to r_range.
            rp_range (list): divergence range to calculate.
            rp_pts (int): points number to rp_range.
            direction (str): direction phase space.

        Returns:
            numpy array: Brilliance.
        """
        spectra_calc: SpectraInterface = copy.deepcopy(self)
        spectra_calc.calc.source_type = source.source_type
        spectra_calc.calc.method = spectra_calc.calc.CalcConfigs.Method.wigner
        if direction == 'vertical':
            spectra_calc.calc.indep_var = (
                spectra_calc.calc.CalcConfigs.Variable.mesh_yyp
            )
        else:
            spectra_calc.calc.indep_var = (
                spectra_calc.calc.CalcConfigs.Variable.mesh_xxp
            )
        spectra_calc.calc.output_type = (
            spectra_calc.calc.CalcConfigs.Output.phasespace
        )

        spectra_calc.calc.period = source.period
        spectra_calc.calc.length = source.source_length

        if direction == 'vertical':
            spectra_calc.calc.y_range = r_range
            spectra_calc.calc.yp_range = rp_range
            spectra_calc.calc.y_nr_pts = r_pts
            spectra_calc.calc.yp_nr_pts = rp_pts
        else:
            spectra_calc.calc.x_range = r_range
            spectra_calc.calc.xp_range = rp_range
            spectra_calc.calc.x_nr_pts = r_pts
            spectra_calc.calc.xp_nr_pts = rp_pts

        if source.polarization == 'hp':
            spectra_calc.calc.ky = target_k
        elif source.polarization == 'vp':
            spectra_calc.calc.kx = target_k
        else:
            spectra_calc.calc.kx = target_k / _np.sqrt(
                1 + source.fields_ratio**2
            )
            spectra_calc.calc.ky = spectra_calc.calc.kx * source.fields_ratio

        spectra_calc.calc.target_harmonic = n_harmonic
        spectra_calc.calc.set_config()
        spectra_calc.calc.run_calculation()
        brilliance = spectra_calc.calc.brilliance.reshape(rp_pts, r_pts)
        del spectra_calc
        return brilliance

    def _parallel_calc_div_size(self, args):
        (
            source,
            target_k,
            n_harmonic,
            r_range,
            r_pts,
            rp_range,
            rp_pts,
            direction,
        ) = args

        if not _np.isnan(target_k):
            brilliance = self.calc_proj_brilliance_with_phasespace(
                source=source,
                target_k=target_k,
                n_harmonic=n_harmonic,
                r_range=[r_range[0], r_range[-1]],
                r_pts=r_pts,
                rp_range=[rp_range[0], rp_range[-1]],
                rp_pts=rp_pts,
                direction=direction,
            )
            brilliance_proj_rp = _np.sum(brilliance, axis=1)
            brilliance_proj_r = _np.sum(brilliance, axis=0)

            rp_div = self.calc_rms(rp_range, brilliance_proj_rp)
            r_size = self.calc_rms(r_range, brilliance_proj_r)

            return rp_div, r_size
        else:
            return _np.nan, _np.nan

    def calc_numerical_div_size_wigner(
        self,
        source,
        emax=20e3,
        e_pts=101,
        direction='vertical',
    ):
        """Calc numerical Divergence and Size of Light Beam.

        Args:
            source: source light.
            emax (float): Máx energy range.
            e_pts (int): points number to energy range.
            direction (str): direction phase space "vertical" or "horizontal".

        Returns:
            div_size (numpy array): Div. at 1th pos. and Size at 2nd pos.
            energies (numpy array).
        """
        if source.min_gap == 0:
            kmax_source = source.calc_max_k(self.accelerator)
        else:
            beff = source.get_beff(source.min_gap / source.period)
            kmax_source = source.undulator_b_to_k(b=beff, period=source.period)

        # Automatic range adjust
        r_lim = 0.01
        rp_lim = 0.01

        r_pts = 31
        rp_pts = 31
        dif_r = 1
        dif_rp = 1
        while dif_r > 0.01 or dif_rp > 0.01:
            brilliance = self.calc_proj_brilliance_with_phasespace(
                source=source,
                target_k=kmax_source,
                n_harmonic=1,
                r_range=[-r_lim, r_lim],
                r_pts=r_pts,
                rp_range=[-rp_lim, rp_lim],
                rp_pts=rp_pts,
                direction=direction,
            )

            brilliance_r = _np.sum(brilliance, axis=0)
            brilliance_r = brilliance_r / _np.max(brilliance_r)
            dif_r = brilliance_r[0]
            if dif_r > 0.01:
                r_lim *= 1.05 + 2 * dif_r

            brilliance_rp = _np.sum(brilliance, axis=1)
            brilliance_rp = brilliance_rp / _np.max(brilliance_rp)
            dif_rp = brilliance_rp[0]
            if dif_rp > 0.01:
                rp_lim *= 1.05 + 2 * dif_rp

        r_lim *= 1.05
        rp_lim *= 1.05
        r_range = _np.linspace(-r_lim, r_lim, r_pts)
        rp_range = _np.linspace(-rp_lim, rp_lim, rp_pts)

        fundamental_energy = source.get_harmonic_energy(
            1, self.accelerator.gamma, 0, source.period, kmax_source
        )
        emin = fundamental_energy
        energies = _np.linspace(emin, emax, e_pts)

        # Arglist Assembly
        arglist = []
        for j, energy in enumerate(energies):
            n = int(energy / fundamental_energy)
            if n > 0:
                n_harmonic = n - 1 if n % 2 == 0 else n
            else:
                n_harmonic = 1
            target_k = source.calc_k_target(
                self.accelerator.gamma, n_harmonic, source.period, energy
            )
            arglist += [
                (
                    source,
                    target_k,
                    n_harmonic,
                    r_range,
                    r_pts,
                    rp_range,
                    rp_pts,
                    direction,
                )
            ]

        # Calc Divergence and Size
        num_process = multiprocessing.cpu_count()
        data = []
        with multiprocessing.Pool(processes=num_process - 1) as parallel:
            data = parallel.map(self._parallel_calc_div_size, arglist)
        div_size = _np.array(data)
        return div_size, energies

    def calc_degree_polarization(
        self,
        source,
        slit_shape: str = 'retslit',
        slit_acceptance: tuple = (0.060, 0.060),
        distance_from_source: float = 30,
        energy_range: tuple = (0, 20e3),
        kmin: float = 0.1,
        k_nr_pts: int = 41,
    ):
        """Degree Polarization Function.

        Args:
            source: light source.
            slit_shape (str): slit shape "retslit" or "circslit".
                Defaults to "retslit".
            slit_position (tuple): slit position [mrad].
                Defaults to (0,0).
            slit_acceptance (tuple): slit acceptance [mrad].
                Defaults to (0.060, 0.060).
            distance_from_source (float): distance from source [m].
                Defaults to 30.
            energy_range (tuple): energy range to calculate [eV].
                Defaults to (0, 20e3).
            kmin (float): min deflection parameter
            k_nr_pts (float): k points number
        Returns:
            Tuple of three elements.
                first element (numpy array): energies
                second element (numpy array): degree linear polarization.
                third element (numpy array): degree circular polarization.
                fourth element (numpy array): degree linear 45 polarization.
        """
        spectra_calc: SpectraInterface = copy.deepcopy(self)
        if source.source_type != 'bendingmagnet':
            if source.min_gap != 0:
                beff = source.get_beff(source.min_gap / source.period)
                kmax = source.undulator_b_to_k(b=beff, period=source.period)
            else:
                kmax = source.calc_max_k(spectra_calc.accelerator)
            kmax = source.calc_max_k(spectra_calc.accelerator)
            fst_energy = source.get_harmonic_energy(
                n=1,
                gamma=spectra_calc.accelerator.gamma,
                theta=0,
                period=source.period,
                k=kmax,
            )
            n = int(energy_range[1] / fst_energy)
            n = n + 1 if n % 2 == 0 else n
            harmonic_range = (1, n)
            if source.source_type == 'wiggler':
                spectra_calc.calc.source_type = source.source_type
                spectra_calc.calc.method = (
                    spectra_calc.calc.CalcConfigs.Method.far_field
                )
                spectra_calc.calc.indep_var = (
                    spectra_calc.calc.CalcConfigs.Variable.k
                )
                spectra_calc.calc.output_type = (
                    spectra_calc.calc.CalcConfigs.Output.flux
                )
                spectra_calc.calc.period = source.period
                spectra_calc.calc.slit_shape = slit_shape
                spectra_calc.calc.slit_acceptance = slit_acceptance
                spectra_calc.calc.k_range = [kmin, kmax]
                spectra_calc.calc.k_nr_pts = k_nr_pts
                spectra_calc.calc.harmonic_range = harmonic_range
            else:
                spectra_calc.calc._add_phase_errors = source.add_phase_errors
                spectra_calc.calc._use_recovery_params = (
                    source.use_recovery_params
                )
                spectra_calc.calc.output_type = (
                    spectra_calc.calc.CalcConfigs.Output.flux
                )
                spectra_calc.calc.method = (
                    spectra_calc.calc.CalcConfigs.Method.far_field
                )
                spectra_calc.calc.indep_var = (
                    spectra_calc.calc.CalcConfigs.Variable.k
                )
                spectra_calc.calc.source_type = source.source_type
                spectra_calc.calc.slit_shape = slit_shape
                spectra_calc.calc.period = source.period
                spectra_calc.calc.slit_shape = slit_shape
                spectra_calc.calc.slit_acceptance = slit_acceptance
                spectra_calc.calc.k_range = [kmin, kmax]
                spectra_calc.calc.k_nr_pts = k_nr_pts
                spectra_calc.calc.harmonic_range = harmonic_range
                if source.polarization == 'hp':
                    spectra_calc.calc.source_type = (
                        spectra_calc.calc.SourceType.horizontal_undulator
                    )
                    spectra_calc.calc.ky = kmax
                elif source.polarization == 'vp':
                    spectra_calc.calc.source_type = (
                        spectra_calc.calc.SourceType.vertical_undulator
                    )
                    spectra_calc.calc.kx = kmax
                else:
                    spectra_calc.calc.source_type = (
                        spectra_calc.calc.SourceType.elliptic_undulator
                    )
                    spectra_calc.calc.kx = kmax / _np.sqrt(
                        1 + source.fields_ratio**2
                    )
                    spectra_calc.calc.ky = (
                        spectra_calc.calc.kx * source.fields_ratio
                    )
                spectra_calc.calc.slit_acceptance = slit_acceptance
        else:
            b = source.b_peak
            spectra_calc.calc.source_type = source.source_type
            spectra_calc.calc.method = (
                spectra_calc.calc.CalcConfigs.Method.far_field
            )
            spectra_calc.calc.indep_var = (
                spectra_calc.calc.CalcConfigs.Variable.energy
            )
            spectra_calc.calc.output_type = (
                spectra_calc.calc.CalcConfigs.Output.flux
            )
            spectra_calc.calc.slit_shape = slit_shape
            spectra_calc.calc.slit_acceptance = slit_acceptance
            spectra_calc.calc.energy_range = energy_range
            spectra_calc.calc.energy_step = 1
            spectra_calc.calc.by_peak = b
        spectra_calc.calc.distance_from_source = distance_from_source
        spectra_calc.calc.length = source.source_length
        spectra_calc.calc.set_config()
        spectra_calc.calc.run_calculation()
        energies = spectra_calc.calc.energies
        degree_pl = spectra_calc.calc._pl
        degree_pc = spectra_calc.calc._pc
        degree_pl45 = spectra_calc.calc._pl45
        del spectra_calc
        return energies, degree_pl, degree_pc, degree_pl45

    def calc_power_density(
        self,
        source,
        x_range: tuple = (-3, 3),
        x_nr_pts: int = 501,
        y_range: tuple = (-3, 3),
        y_nr_pts: int = 501,
        distance_from_source: float = 30,
        current: float = 350,
    ):
        """Power Density Function.

        Args:
            source: light source.
            x_range (tuple): x range to calculate power dentity [mrad].
                Defaults to (-3,3).
            x_nr_pts (int): x number points of x_range.
                Defaults to 501.
            y_range (tuple): y range to calculate power dentity [mrad].
                Defaults to (-3, 3).
            y_nr_pts (int): y number points of y_range.
                Defaults to 501.
            distance_from_source (float): distance from source [m].
                Defaults to 30.
            current (float): current [mA].
                Defaults to 350.

        Return:
            Power densities (Numpy array)
        """
        spectra_calc: SpectraInterface = copy.deepcopy(self)
        spectra_calc.accelerator.current = current
        spectra_calc.calc.source_type = source.source_type
        spectra_calc.calc.method = (
            spectra_calc.calc.CalcConfigs.Method.near_field
        )
        spectra_calc.calc.indep_var = (
            spectra_calc.calc.CalcConfigs.Variable.mesh_xy
        )
        spectra_calc.calc.output_type = (
            spectra_calc.calc.CalcConfigs.Output.power_density
        )
        if source.source_type != 'bendingmagnet':
            kmax = source.calc_max_k(spectra_calc.accelerator)
            if source.polarization == 'hp':
                spectra_calc.calc.source_type = (
                    spectra_calc.calc.SourceType.horizontal_undulator
                )
                spectra_calc.calc.ky = kmax
            elif source.polarization == 'vp':
                spectra_calc.calc.source_type = (
                    spectra_calc.calc.SourceType.vertical_undulator
                )
                spectra_calc.calc.kx = kmax
            else:
                spectra_calc.calc.source_type = (
                    spectra_calc.calc.SourceType.elliptic_undulator
                )
                spectra_calc.calc.kx = kmax / _np.sqrt(
                    1 + source.fields_ratio**2
                )
                spectra_calc.calc.ky = (
                    spectra_calc.calc.kx * source.fields_ratio
                )
            spectra_calc.calc.period = source.period
            spectra_calc.calc.length = source.source_length
        else:
            spectra_calc.calc.by_peak = source.b_peak
            spectra_calc.calc.length = 0.05
        spectra_calc.calc.distance_from_source = distance_from_source
        spectra_calc.calc.x_range = x_range
        spectra_calc.calc.y_range = y_range
        spectra_calc.calc.x_nr_pts = x_nr_pts
        spectra_calc.calc.y_nr_pts = y_nr_pts
        spectra_calc.calc.set_config()
        spectra_calc.calc.run_calculation()
        power_densities = spectra_calc.calc.power_density
        del spectra_calc
        return power_densities

    def calc_partial_power(
        self,
        source,
        slit_shape: str = 'retslit',
        slit_position: tuple = (0, 0),
        slit_acceptance: tuple = (0.060, 0.060),
        distance_from_source: float = 30,
        current: float = 350,
    ):
        """Partial Power Function.

        Args:
            source: light source.
            slit_shape (str): slit shape "retslit" or "circslit".
                Defaults to "retslit"
            slit_position (tuple): slit position [mrad].
                Defaults to (0,0)
            slit_acceptance (tuple): slit acceptance [mrad].
                Defaults to (0.060, 0.060)
            distance_from_source (float): distance from source [m].
                Defaults to 30.
            current (float): current [mA].
                Defaults to 350.

        Return:
            Partial Power (float)
        """
        spectra_calc: SpectraInterface = copy.deepcopy(self)
        spectra_calc.accelerator.current = current
        spectra_calc.calc.source_type = source.source_type
        spectra_calc.calc.method = (
            spectra_calc.calc.CalcConfigs.Method.fixedpoint_near_field
        )
        spectra_calc.calc.indep_var = (
            spectra_calc.calc.CalcConfigs.Variable.energy
        )
        spectra_calc.calc.output_type = (
            spectra_calc.calc.CalcConfigs.Output.power
        )
        spectra_calc.calc.slit_shape = slit_shape

        spectra_calc.calc.target_energy = 0
        if source.source_type != 'bendingmagnet':
            kmax = source.calc_max_k(spectra_calc.accelerator)
            if source.polarization == 'hp':
                spectra_calc.calc.source_type = (
                    spectra_calc.calc.SourceType.horizontal_undulator
                )
                spectra_calc.calc.ky = kmax
            elif source.polarization == 'vp':
                spectra_calc.calc.source_type = (
                    spectra_calc.calc.SourceType.vertical_undulator
                )
                spectra_calc.calc.kx = kmax
            else:
                spectra_calc.calc.source_type = (
                    spectra_calc.calc.SourceType.elliptic_undulator
                )
                spectra_calc.calc.kx = kmax / _np.sqrt(
                    1 + source.fields_ratio**2
                )
                spectra_calc.calc.ky = (
                    spectra_calc.calc.kx * source.fields_ratio
                )
            spectra_calc.calc.period = source.period
            spectra_calc.calc.length = source.source_length
        else:
            spectra_calc.calc.by_peak = source.b_peak
            spectra_calc.calc.length = 0.05

        spectra_calc.calc.distance_from_source = distance_from_source
        spectra_calc.calc.observation_angle = slit_position
        spectra_calc.calc.slit_acceptance = slit_acceptance

        spectra_calc.calc.set_config()
        spectra_calc.calc.run_calculation()
        partial_power = spectra_calc.calc.power
        del spectra_calc
        return partial_power

    def calc_flux_distribution_2d(
        self,
        source,
        target_energy: float = 12e3,
        target_k: float = 1.2,
        x_range: tuple = (-3, 3),
        x_nr_pts: int = 401,
        y_range: tuple = (-3, 3),
        y_nr_pts: int = 401,
        distance_from_source: float = 30,
    ):
        """Flux Distribution 2D.

        Args:
            source: light source.
            target_energy (float): target energy [eV].
            target_k (float): k deflection parameter.
            x_range (tuple): x range to calculate power dentity [mrad].
                Defaults to (-3, 3).
            x_nr_pts (int): x number points of x_range.
                Defaults to 401.
            y_range (tuple): y range to calculate power dentity [mrad].
                Defaults to (-3, 3).
            y_nr_pts (int): y number points of y_range.
                Defaults to 401.
            distance_from_source (float): distance from source [m].
                Defaults to 30.

        Return:
            Flux distribuition 2D (Numpy array)
        """
        spectra_calc: SpectraInterface = copy.deepcopy(self)
        spectra_calc.calc.source_type = source.source_type
        spectra_calc.calc.output_type = (
            spectra_calc.calc.CalcConfigs.Output.flux_density
        )
        spectra_calc.calc.indep_var = (
            spectra_calc.calc.CalcConfigs.Variable.mesh_xy
        )
        if source.source_type != 'bendingmagnet':
            target_k = target_k
            source_polarization = source.polarization
            spectra_calc.calc.period = source.period
            spectra_calc.calc.length = source.source_length

            if source_polarization == 'hp':
                spectra_calc.calc.ky = target_k
            elif source_polarization == 'vp':
                spectra_calc.calc.kx = target_k
            else:
                spectra_calc.calc.kx = target_k / _np.sqrt(
                    1 + source.fields_ratio**2
                )
                spectra_calc.calc.ky = (
                    spectra_calc.calc.kx * source.fields_ratio
                )
            spectra_calc.calc.method = (
                spectra_calc.calc.CalcConfigs.Method.far_field
            )
        else:
            spectra_calc.calc.by = source.b_peak
            spectra_calc.calc.method = (
                spectra_calc.calc.CalcConfigs.Method.near_field
            )
        spectra_calc.calc.x_nr_pts = x_nr_pts
        spectra_calc.calc.y_nr_pts = y_nr_pts
        spectra_calc.calc.x_range = [
            x_range[0],
            x_range[1],
        ]
        spectra_calc.calc.y_range = [
            y_range[0],
            y_range[1],
        ]
        spectra_calc.calc.distance_from_source = distance_from_source
        spectra_calc.calc.target_energy = target_energy
        spectra_calc.calc.set_config()
        spectra_calc.calc.run_calculation()
        flux_distribuition = spectra_calc.calc.flux
        if source.source_type != 'bendingmagnet':
            if source.use_recovery_params and source.add_phase_errors:
                fundamental_energy = source.get_harmonic_energy(
                    1,
                    spectra_calc.accelerator.gamma,
                    0,
                    source.period,
                    source.calc_max_k(spectra_calc.accelerator),
                )
                target_harmonic = int(target_energy / fundamental_energy)
                target_harmonic = (
                    target_harmonic - 1
                    if target_harmonic % 2 == 0
                    else target_harmonic
                )
                target_harmonic = (
                    1 if target_harmonic <= 0 else target_harmonic
                )
                spectra_calc.use_recovery_params = True
                flux_distribuition = spectra_calc.apply_phase_error_matrix(
                    values=flux_distribuition,
                    harm=target_harmonic,
                    rec_param=spectra_calc.use_recovery_params,
                )
        del spectra_calc
        return flux_distribuition

    def calc_partial_flux(
        self,
        source,
        target_energy: float = 12e3,
        target_k: float = 1.2,
        slit_shape: str = 'retslit',
        slit_acceptance: tuple = (0.06, 0.060),
        slit_position: tuple = (0, 0),
        distance_from_source: float = 30,
    ):
        """Partial Flux with fixedpoint method.

        Args:
            source: light source.
            target_energy (float): target energy [eV].
            target_k (float): k deflection parameter.
            slit_shape (str): slit shape "retslit" or "circslit".
                Defaults to "retslit".
            slit_acceptance (tuple): slit acceptance [mrad].
                Defaults to (0.060, 0.060)
            slit_position (tuple): slit position [mrad].
                Defaults to (0, 0)
            distance_from_source (float): distance from source [m].
                Defaults to 30.

        Return:
            Partial Flux value (float)
        """
        spectra_calc: SpectraInterface = copy.deepcopy(self)
        spectra_calc.calc.source_type = source.source_type
        spectra_calc.calc.indep_var = (
            spectra_calc.calc.CalcConfigs.Variable.energy
        )
        spectra_calc.calc.method = (
            spectra_calc.calc.CalcConfigs.Method.fixedpoint_near_field
        )
        spectra_calc.calc.output_type = (
            spectra_calc.calc.CalcConfigs.Output.flux
        )

        spectra_calc.calc.slit_shape = slit_shape
        spectra_calc.calc.slit_acceptance = [
            slit_acceptance[0],
            slit_acceptance[1],
        ]
        spectra_calc.calc.observation_angle = [
            slit_position[0],
            slit_position[1],
        ]

        if source.source_type != 'bendingmagnet':
            target_k = target_k
            source_polarization = source.polarization
            spectra_calc.calc.period = source.period
            spectra_calc.calc.length = source.source_length

            if source_polarization == 'hp':
                spectra_calc.calc.ky = target_k
            elif source_polarization == 'vp':
                spectra_calc.calc.kx = target_k
            else:
                spectra_calc.calc.kx = target_k / _np.sqrt(
                    1 + source.fields_ratio**2
                )
                spectra_calc.calc.ky = (
                    spectra_calc.calc.kx * source.fields_ratio
                )
        else:
            spectra_calc.calc.by = source.b_peak

        spectra_calc.calc.distance_from_source = distance_from_source
        spectra_calc.calc.target_energy = target_energy

        spectra_calc.calc.set_config()
        spectra_calc.calc.run_calculation()
        flux_total = spectra_calc.calc.flux
        if source.source_type != 'bendingmagnet':
            if source.use_recovery_params and source.add_phase_errors:
                fundamental_energy = source.get_harmonic_energy(
                    1,
                    spectra_calc.accelerator.gamma,
                    0,
                    source.period,
                    source.calc_max_k(spectra_calc.accelerator),
                )
                target_harmonic = int(target_energy / fundamental_energy)
                target_harmonic = (
                    target_harmonic - 1
                    if target_harmonic % 2 == 0
                    else target_harmonic
                )
                target_harmonic = (
                    1 if target_harmonic <= 0 else target_harmonic
                )
                spectra_calc.use_recovery_params = True
                flux_total = spectra_calc.apply_phase_error_matrix(
                    values=flux_total,
                    harm=target_harmonic,
                    rec_param=spectra_calc.use_recovery_params,
                )
        del spectra_calc
        return flux_total

    def plot_brilliance_curve(  # noqa: C901
        self,
        process_curves=True,
        superp_value=250,
        title='Brilliance curves',
        xscale='linear',
        yscale='log',
        xlim=[],
        ylim=[],
        linewidth=3,
        savefig=False,
        figsize=(4.5, 3.0),
        figname='brill.png',
        dpi=300,
        legend_fs=10,
        legend_properties=True,
    ):
        """Plot brilliance curves.

        Args:
            process_curves (bool, optional): If true energy superposition will
             be processed. Defaults to True.
            superp_value (int, optional): Desired value of energy
             superposition. Defaults to 250.
            title (str, optional): Plot title.
            xscale (str, optional): xscale axis
             xscale. Defalts to 'linear'.
            yscale (str, optional): yscale axis
             yscale. Defalts to 'log'.
        """
        if self._flag_brill_processed:
            process_curves = False
        energies = list()
        brilliances = list()
        if process_curves is True:
            self._flag_brill_processed = True
            for i, source in enumerate(self.sources):
                if (
                    source.source_type != 'wiggler'
                    and source.source_type != 'bendingmagnet'
                ):
                    input_brilliance = self.brilliances[i][:, :]
                    input_energies = self.energies[i][:, :]
                    if input_energies.shape[0] > 1:
                        energies_, brilliance = (
                            self.calc.process_brilliance_curve(
                                input_energies,
                                input_brilliance,
                                superp_value=superp_value,
                            )
                        )
                    else:
                        input_brilliance_b = input_brilliance[0]
                        input_energies_b = input_energies[0]
                        idx = _np.argsort(input_energies_b)
                        input_energies_b = input_energies_b[idx]
                        input_brilliance_b = input_brilliance_b[idx]
                        energies_ = _np.linspace(
                            _np.min(input_energies_b),
                            _np.max(input_energies_b),
                            2001,
                        )
                        brilliance = _np.interp(
                            energies_, input_energies_b, input_brilliance_b
                        )
                        energies_ = _np.reshape(
                            energies_, (1, _np.shape(energies_)[0])
                        )
                        brilliance = _np.reshape(
                            brilliance, (1, _np.shape(brilliance)[0])
                        )
                else:
                    input_brilliance = _np.array(
                        self.brilliances[i], dtype=float
                    )
                    input_energies = _np.array(self.energies[i], dtype=float)
                    energies_ = _np.linspace(
                        _np.min(input_energies), _np.max(input_energies), 2001
                    )
                    brilliance = _np.interp(
                        energies_, input_energies, input_brilliance
                    )
                    energies_ = _np.reshape(
                        energies_, (1, _np.shape(energies_)[0])
                    )
                    brilliance = _np.reshape(
                        brilliance, (1, _np.shape(brilliance)[0])
                    )

                energies.append(energies_)
                brilliances.append(brilliance)
            energies = _np.array(energies, dtype=object)
            brilliances = _np.array(brilliances, dtype=object)
            self._energies = energies
            self._brilliances = brilliances

        _plt.figure(figsize=figsize)
        colorlist = ['C' + str(i) for i, value in enumerate(self.sources)]
        for i, source in enumerate(self.sources):
            color = colorlist[i]
            if source.source_type == 'bendingmagnet':
                label = source.label
            else:
                label = source.label
                if legend_properties:
                    label += ', λ = {:.1f} mm'.format(source.period)
                    label += ', L = {:.1f} m'.format(source.source_length)
            for j in _np.arange(self.energies[i].shape[0]):
                if j == 0:
                    _plt.plot(
                        1e-3 * self.energies[i][j, :],
                        self.brilliances[i][j, :],
                        color=color,
                        linewidth=linewidth,
                        alpha=0.9,
                        label=label,
                    )
                else:
                    _plt.plot(
                        1e-3 * self.energies[i][j, :],
                        self.brilliances[i][j, :],
                        color=color,
                        linewidth=linewidth,
                        alpha=0.9,
                    )

        _plt.yscale(yscale)
        _plt.xscale(xscale)

        if xlim:
            _plt.xlim(xlim[0], xlim[1])
        if ylim:
            _plt.ylim(ylim[0], ylim[1])

        _plt.xlabel('Energy [keV]')
        _plt.ylabel('Brilliance [ph/s/0.1%/mm²/mrad²/100mA]')
        _plt.title(title)

        _plt.minorticks_on()
        _plt.tick_params(
            which='both', axis='both', direction='in', top=True, right=True
        )
        _plt.grid(which='major', alpha=0.4)
        _plt.grid(which='minor', alpha=0.2)

        _plt.legend(fontsize=legend_fs)
        _plt.tight_layout()

        if savefig:
            _plt.savefig(figname, dpi=dpi)
        else:
            _plt.show()

    def plot_flux_curve(  # noqa: C901
        self,
        process_curves=True,
        superp_value=250,
        title='Flux curves',
        xscale='linear',
        yscale='log',
        xlim=[],
        ylim=[],
        linewidth=3,
        savefig=False,
        figsize=(4.5, 3.0),
        figname='flux.png',
        dpi=300,
        legend_fs=10,
        legend_properties=True,
    ):
        """Plot flux curves.

        Args:
            process_curves (bool, optional): If true energy superposition will
             be processed. Defaults to True.
            superp_value (int, optional): Desired value of energy
             superposition. Defaults to 250.
            title (str, optional): Plot title.
            xscale (str, optional): xscale axis
             xscale. Defaults to 'linear'.
            yscale (str, optional): yscale axis
             yscale. Defaults to 'log'.
            xlim (list, optional): xlim axes.
            ylim (list, optional): ylim axes.
            linewidth (int, optional): linewidth.
            savefig (bool, optional): save fig.
            figname (str, optional): figname.
             figname. Defaults to 'flux.png'.
            dpi (int, optional): dpi figure.
             dpi. Defaults to 300.
            legend_fs (int, optional): legend font size.
             legend_fs. Defaults to 10.
            legend_properties (bool, optional): lengend properties.
             legend_properties. Defaults to True

        """
        if self._flag_flux_processed:
            process_curves = False
        energies = list()
        fluxes = list()
        if process_curves is True:
            self._flag_flux_processed = True
            for i, source in enumerate(self.sources):
                if (
                    source.source_type != 'wiggler'
                    and source.source_type != 'bendingmagnet'
                ):
                    input_flux = self.fluxes[i][:, :]
                    input_energies = self.energies[i][:, :]
                    if input_energies.shape[0] > 1:
                        energies_, flux = self.calc.process_brilliance_curve(
                            input_energies,
                            input_flux,
                            superp_value=superp_value,
                        )
                    else:
                        input_flux_b = input_flux[0]
                        input_energies_b = input_energies[0]
                        idx = _np.argsort(input_energies_b)
                        input_energies_b = input_energies_b[idx]
                        input_flux_b = input_flux_b[idx]
                        energies_ = _np.linspace(
                            _np.min(input_energies_b),
                            _np.max(input_energies_b),
                            2001,
                        )
                        flux = _np.interp(
                            energies_, input_energies_b, input_flux_b
                        )
                        energies_ = _np.reshape(
                            energies_, (1, _np.shape(energies_)[0])
                        )
                        flux = _np.reshape(flux, (1, _np.shape(flux)[0]))
                else:
                    input_flux = _np.array(self.fluxes[i], dtype=float)
                    input_energies = _np.array(self.energies[i], dtype=float)
                    energies_ = _np.linspace(
                        _np.min(input_energies), _np.max(input_energies), 2001
                    )
                    flux = _np.interp(energies_, input_energies, input_flux)
                    energies_ = _np.reshape(
                        energies_, (1, _np.shape(energies_)[0])
                    )
                    flux = _np.reshape(flux, (1, _np.shape(flux)[0]))

                energies.append(energies_)
                fluxes.append(flux)
            energies = _np.array(energies, dtype=object)
            fluxes = _np.array(fluxes, dtype=object)
            self._energies = energies
            self._fluxes = fluxes

        _plt.figure(figsize=figsize)
        colorlist = ['C' + str(i) for i, value in enumerate(self.sources)]
        for i, source in enumerate(self.sources):
            color = colorlist[i]
            if source.source_type == 'bendingmagnet':
                label = source.label
            else:
                label = source.label
                if legend_properties:
                    label += ', λ = {:.1f} mm'.format(source.period)
                    label += ', L = {:.1f} m'.format(source.source_length)
            for j in _np.arange(self.energies[i].shape[0]):
                if j == 0:
                    _plt.plot(
                        1e-3 * self.energies[i][j, :],
                        self.fluxes[i][j, :],
                        color=color,
                        linewidth=linewidth,
                        alpha=0.9,
                        label=label,
                    )
                else:
                    _plt.plot(
                        1e-3 * self.energies[i][j, :],
                        self.fluxes[i][j, :],
                        color=color,
                        linewidth=linewidth,
                        alpha=0.9,
                    )

        _plt.yscale(yscale)
        _plt.xscale(xscale)

        if xlim:
            _plt.xlim(xlim[0], xlim[1])
        if ylim:
            _plt.ylim(ylim[0], ylim[1])

        _plt.xlabel('Energy [keV]')
        _plt.ylabel('Flux [ph/s/0.1%/100mA]')
        _plt.title(title)

        _plt.minorticks_on()
        _plt.tick_params(
            which='both', axis='both', direction='in', top=True, right=True
        )
        _plt.grid(which='major', alpha=0.4)
        _plt.grid(which='minor', alpha=0.2)

        _plt.legend(fontsize=legend_fs)
        _plt.tight_layout()

        if savefig:
            _plt.savefig(figname, dpi=dpi)
        else:
            _plt.show()

    def plot_flux_density_matrix(
        self,
        title=None,
        clim=(None, None),
        cscale='linear',
        savefig=False,
        figsize=(5, 4),
        figname='flux_density_matrix.png',
        dpi=400,
    ):
        """Plot Flux Density Matrix (period x length).

        Args:
            title (str, optional): Plot title.
            cscale (str, optional): color bar scale
             cscale. Defalts to 'linear'.
            clim (tuple): color bar limits.
             Defaults to (None, None) will take the minimum or/and maximum limit
            savefig (bool, optional): Save Figure
             savefig. Defalts to False.
            figname (str, optional): Figure name
             figname. Defalts to 'flux_density_matrix.png'
            dpi (int, optional): Image resolution
             dpi. Defalts to 400.
            figsize (tuple, optional): Figure size.
             figsize. Defalts to (5, 4)
        """
        # Getting the parameters of the best undulator
        info = self._info_matrix_flux_density[
            _np.argmax(self._flux_density_matrix.ravel())
        ]

        period_number = info[1]
        length_number = info[2]

        # Getting the position of the best flux density
        j = int(
            _np.argmax(self._flux_density_matrix.ravel())
            / len(self._flux_density_matrix[0, :])
        )
        i = _np.argmax(self._flux_density_matrix.ravel()) % len(
            self._flux_density_matrix[0, :]
        )

        # Label creation
        label = 'Target Energy: {:.2f} KeV\n'.format(self._target_energy / 1e3)
        label += 'Best undulator: ({:.2f} mm, {:.2f} m)\n'.format(
            period_number, length_number
        )
        label += 'Flux density: {:.2e} ph/s/mrad²/0.1%/100mA'.format(
            self._flux_density_matrix[j, i]
        )

        fig, ax = _plt.subplots(figsize=figsize)
        ax.set_title(label if title == None else title)
        ax.set_ylabel('Length [m]')
        ax.set_xlabel('Period [mm]')

        vmin = clim[0]
        vmax = clim[1]

        if vmin is None:
            vmin = _np.min(self._flux_density_matrix)
        if vmax is None:
            vmax = _np.max(self._flux_density_matrix)

        step = (
            5
            if cscale == 'linear'
            else int(_np.log10(vmax) - _np.log10(vmin) + 1)
        )
        vmin = vmin if cscale == 'linear' else _np.log10(vmin)
        vmax = vmax if cscale == 'linear' else _np.log10(vmax)
        fm = (
            self._flux_density_matrix
            if cscale == 'linear'
            else _np.log10(self._flux_density_matrix)
        )

        ax.imshow(
            fm,
            extent=[
                self._info_matrix_flux_density[0, 1],
                self._info_matrix_flux_density[-1, 1],
                self._info_matrix_flux_density[0, 2],
                self._info_matrix_flux_density[-1, 2],
            ],
            aspect='auto',
            origin='lower',
            norm=colors.Normalize(vmin=vmin, vmax=vmax)
            if cscale == 'linear'
            else colors.LogNorm(vmin=vmin, vmax=vmax),
        )
        sm = _plt.cm.ScalarMappable(_plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array(fm)
        cbar = fig.colorbar(
            sm,
            ax=ax,
            label='Flux density [ph/s/mrad²/0.1%/100mA]',
            format='%.1e' if cscale == 'linear' else '%.0i',
        )
        cbar.set_ticks(_np.linspace(vmin, vmax, step))
        fig.tight_layout()
        if savefig:
            _plt.savefig(figname, dpi=dpi)
        else:
            _plt.show()

    def plot_flux_matrix(
        self,
        data=None,
        title=None,
        clim=(None, None),
        cscale='linear',
        cmap='viridis',
        savefig=False,
        figsize=(5, 4),
        figname='flux_matrix.png',
        dpi=400,
    ):
        """Plot Flux Matrix (period x length).

        Args:
            title (str, optional): Plot title.
            cscale (str, optional): color bar scale
             cscale. Defalts to 'linear'.
            clim (tuple): color bar limits.
             Defaults to (None, None) will take the minimum or/and maximum limit
            cmap (str): colormap.
            savefig (bool, optional): Save Figure
             savefig. Defalts to False.
            figname (str, optional): Figure name
             figname. Defalts to 'flux_matrix.png'
            dpi (int, optional): Image resolution
             dpi. Defalts to 400.
            figsize (tuple, optional): Figure size.
             figsize. Defalts to (5, 4)
        """
        # Getting the parameters of the best undulator
        flux_matrix = data[0]
        info_matrix = data[1]
        info = info_matrix[_np.argmax(flux_matrix.ravel())]

        period_number = info[1]
        length_number = info[2]

        # Getting the position of the best brilliance
        j = int(_np.argmax(flux_matrix.ravel()) / len(flux_matrix[0, :]))
        i = _np.argmax(flux_matrix.ravel()) % len(flux_matrix[0, :])

        # Label creation
        label = 'Target Energy: {:.2f} KeV\n'.format(self._target_energy / 1e3)
        label += 'Best undulator: ({:.2f} mm, {:.2f} m)\n'.format(
            period_number, length_number
        )
        label += 'Flux: {:.2e} ph/s/0.1%/100mA'.format(flux_matrix[j, i])

        fig, ax = _plt.subplots(figsize=figsize)
        ax.set_title(label if title == None else title)
        ax.set_ylabel('Length [m]')
        ax.set_xlabel('Period [mm]')

        vmin = clim[0]
        vmax = clim[1]

        if vmin is None:
            vmin = _np.min(flux_matrix)
        if vmax is None:
            vmax = _np.max(flux_matrix)

        step = (
            5
            if cscale == 'linear'
            else int(_np.log10(vmax) - _np.log10(vmin) + 1)
        )
        vmin = vmin if cscale == 'linear' else _np.log10(vmin)
        vmax = vmax if cscale == 'linear' else _np.log10(vmax)
        fm = flux_matrix if cscale == 'linear' else _np.log10(flux_matrix)

        ax.imshow(
            fm,
            extent=[
                info_matrix[0, 1],
                info_matrix[-1, 1],
                info_matrix[0, 2],
                info_matrix[-1, 2],
            ],
            aspect='auto',
            origin='lower',
            cmap=cmap,
            norm=colors.Normalize(vmin=vmin, vmax=vmax)
            if cscale == 'linear'
            else colors.LogNorm(vmin=vmin, vmax=vmax),
        )
        sm = _plt.cm.ScalarMappable(
            _plt.Normalize(vmin=vmin, vmax=vmax), cmap=cmap
        )
        sm.set_array(fm)
        cbar = fig.colorbar(
            sm,
            ax=ax,
            label='Flux [ph/s/0.1%/100mA]',
            format='%.1e' if cscale == 'linear' else '%.0i',
            cmap=cmap,
        )
        cbar.set_ticks(_np.linspace(vmin, vmax, step))
        fig.tight_layout()
        if savefig:
            _plt.savefig(figname, dpi=dpi)
        else:
            _plt.show()

    def plot_brilliance_matrix(
        self,
        data=None,
        title=None,
        clim=(None, None),
        cscale='linear',
        cmap='viridis',
        savefig=False,
        figsize=(5, 4),
        figname='brilliance_matrix.png',
        dpi=400,
    ):
        """Plot Brilliance Matrix (period x length).

        Args:
            title (str, optional): Plot title.
            cscale (str, optional): color bar scale
             cscale. Defalts to 'linear'.
            clim (tuple): color bar limits.
             Defaults to (None, None) will take the minimum or/and maximum limit
            savefig (bool, optional): Save Figure
             savefig. Defalts to False.
            figname (str, optional): Figure name
             figname. Defalts to 'brilliance_matrix.png'
            dpi (int, optional): Image resolution
             dpi. Defalts to 400.
            figsize (tuple, optional): Figure size.
             figsize. Defalts to (5, 4)
        """
        # Getting the parameters of the best undulator
        brilliance_matrix = data[0]
        info_matrix = data[1]
        info = info_matrix[_np.argmax(brilliance_matrix.ravel())]

        period_number = info[1]
        length_number = info[2]

        # Getting the position of the best brilliance
        j = int(
            _np.argmax(brilliance_matrix.ravel())
            / len(brilliance_matrix[0, :])
        )
        i = _np.argmax(brilliance_matrix.ravel()) % len(
            brilliance_matrix[0, :]
        )

        # Label creation
        label = 'Target Energy: {:.2f} KeV\n'.format(self._target_energy / 1e3)
        label += 'Best undulator: ({:.2f} mm, {:.2f} m)\n'.format(
            period_number, length_number
        )
        label += 'Brilliance: {:.2e} ph/s/0.1%/mm²/mrad²/100mA'.format(
            brilliance_matrix[j, i]
        )

        fig, ax = _plt.subplots(figsize=figsize)
        ax.set_title(label if title is None else title)
        ax.set_ylabel('Length [m]')
        ax.set_xlabel('Period [mm]')

        vmin = clim[0]
        vmax = clim[1]

        if vmin is None:
            vmin = _np.min(brilliance_matrix)
        if vmax is None:
            vmax = _np.max(brilliance_matrix)

        step = (
            5
            if cscale == 'linear'
            else int(_np.log10(vmax) - _np.log10(vmin) + 1)
        )
        vmin = vmin if cscale == 'linear' else _np.log10(vmin)
        vmax = vmax if cscale == 'linear' else _np.log10(vmax)
        bm = (
            brilliance_matrix
            if cscale == 'linear'
            else _np.log10(brilliance_matrix)
        )

        ax.imshow(
            bm,
            extent=[
                info_matrix[0, 1],
                info_matrix[-1, 1],
                info_matrix[0, 2],
                info_matrix[-1, 2],
            ],
            aspect='auto',
            origin='lower',
            cmap=cmap,
            norm=colors.Normalize(vmin=vmin, vmax=vmax)
            if cscale == 'linear'
            else colors.LogNorm(vmin=vmin, vmax=vmax),
        )
        sm = _plt.cm.ScalarMappable(
            _plt.Normalize(vmin=vmin, vmax=vmax), cmap=cmap
        )
        sm.set_array(bm)
        cbar = fig.colorbar(
            sm,
            ax=ax,
            label='Brilliance [ph/s/0.1%/mm²/mrad²/100mA]',
            format='%.1e' if cscale == 'linear' else '%.0i',
        )
        cbar.set_ticks(_np.linspace(vmin, vmax, step))
        fig.tight_layout()
        if savefig:
            _plt.savefig(figname, dpi=dpi)
        else:
            _plt.show()

    def plot_total_power_matrix(
        self,
        data: tuple,
        title: str = 'Total Power of Undulators',
        clim: tuple = (None, None),
        cscale: str = 'linear',
        savefig: bool = False,
        figsize: tuple = (5, 4),
        figname: str = 'total_power_matrix.png',
        dpi: int = 400,
    ):
        """Plot Total Power Matrix (period x length).

        Args:
            title (str, optional): Plot title.
            cscale (str, optional): color bar scale
             cscale. Defalts to 'linear'.
            clim (tuple): color bar limits.
             Defaults to (None, None) will take the minimum or/and maximum limit
            savefig (bool, optional): Save Figure
             savefig. Defalts to False.
            figsize (tuple, optional): Figure size.
             figsize. Defalts to (5, 4)
            figname (str, optional): Figure name
             figname. Defalts to 'total_power_matrix.png'
            dpi (int, optional): Image resolution
             dpi. Defalts to 400.
            data (tuple): data especified
             First position 'flux matrix' or 'flux density matrix' or 'brilliance matrix'
             Second position unds matrix
        """
        if data is None:
            raise ValueError("'unds_matrix' parameter has to be defined")

        vmin = clim[0]
        vmax = clim[1]

        info_unds_matrix = data[1]

        current = 100
        ks = info_unds_matrix[:, 0]
        periods = info_unds_matrix[:, 1]
        lengths = info_unds_matrix[:, 2]

        # Calc Fields
        bs = (ks * EMASS * LSPEED * 2 * PI) / (ECHARGE * periods * 1e-3)

        # Calc total power
        const = ((ECHARGE**4) * (self.accelerator.gamma**2)) / (
            12 * PI * VACUUM_PERMITTICITY * (EMASS**2) * (LSPEED**2)
        )
        total_powers = (
            const * (bs**2) * lengths * (current * 1e-3) / (1e3 * ECHARGE)
        )

        pts_period = len(
            _np.where(info_unds_matrix[:, 2] == info_unds_matrix[0, 2])[0]
        )
        pts_length = len(
            _np.where(info_unds_matrix[:, 1] == info_unds_matrix[0, 1])[0]
        )

        total_powers = total_powers.reshape(pts_length, pts_period)

        if vmin is None:
            vmin = _np.min(total_powers)
        if vmax is None:
            vmax = _np.max(total_powers)

        fig, ax = _plt.subplots(figsize=(figsize))
        ax.set_title(title)
        ax.set_ylabel('Length [m]')
        ax.set_xlabel('Period [mm]')

        step = (
            5
            if cscale == 'linear'
            else int(_np.log10(vmax) - _np.log10(vmin) + 1)
        )
        vmin = vmin if cscale == 'linear' else _np.log10(vmin)
        vmax = vmax if cscale == 'linear' else _np.log10(vmax)
        pm = total_powers if cscale == 'linear' else _np.log10(total_powers)

        ax.imshow(
            pm,
            extent=[
                _np.min(periods),
                _np.max(periods),
                _np.min(lengths),
                _np.max(lengths),
            ],
            aspect='auto',
            origin='lower',
            norm=colors.Normalize(vmin=vmin, vmax=vmax)
            if cscale == 'linear'
            else colors.LogNorm(vmin=vmin, vmax=vmax),
        )

        sm = _plt.cm.ScalarMappable(_plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array(pm)

        cbar = fig.colorbar(
            sm,
            ax=ax,
            label='Total Power [kW]',
            format='%.2f' if cscale == 'linear' else '%.0i',
        )
        cbar.set_ticks(_np.linspace(vmin, vmax, step))
        fig.tight_layout()
        if savefig:
            _plt.savefig(figname, dpi=dpi)
        else:
            _plt.show()

    def plot_partial_power_matrix(
        self,
        title: str = 'Partial Power of Undulators',
        clim: tuple = (None, None),
        cscale: str = 'linear',
        savefig: bool = False,
        figsize: tuple = (5, 4),
        figname: str = 'partial_power_matrix.png',
        dpi: int = 400,
        partial_power_matrix=None,
    ):
        """Plot Partial Power Matrix (period x length).

        Args:
            title (str, optional): Plot title.
            cscale (str, optional): color bar scale
             cscale. Defalts to 'linear'.
            clim (tuple): color bar limits.
             Defaults to (None, None) will take the minimum or/and maximum limit
            savefig (bool, optional): Save Figure
             savefig. Defalts to False.
            figsize (tuple, optional): Figure size.
             figsize. Defalts to (5, 4)
            figname (str, optional): Figure name
             figname. Defalts to 'partial_power_matrix.png'
            dpi (int, optional): Image resolution
             dpi. Defalts to 400.
            partial_power_matrix (numpy array): partial power matrix of undulators information to use in calculation
        """
        if partial_power_matrix is None:
            raise ValueError(
                "'partial_power_matrix' parameter has to be defined"
            )

        if self._info_matrix_flux is not None:
            info_unds_matrix = self._info_matrix_flux
        elif self._info_matrix_flux_density is not None:
            info_unds_matrix = self._info_matrix_flux_density
        elif self._info_matrix_brilliance is not None:
            info_unds_matrix = self._info_matrix_brilliance

        periods = info_unds_matrix[:, 1]
        lengths = info_unds_matrix[:, 2]

        vmin = clim[0]
        vmax = clim[1]

        if vmin is None:
            vmin = _np.min(partial_power_matrix)
        if vmax is None:
            vmax = _np.max(partial_power_matrix)

        fig, ax = _plt.subplots(figsize=(figsize))
        ax.set_title(title)
        ax.set_ylabel('Length [m]')
        ax.set_xlabel('Period [mm]')

        step = (
            5
            if cscale == 'linear'
            else int(_np.log10(vmax) - _np.log10(vmin) + 1)
        )
        vmin = vmin if cscale == 'linear' else _np.log10(vmin)
        vmax = vmax if cscale == 'linear' else _np.log10(vmax)
        partial_power_matrix = (
            partial_power_matrix
            if cscale == 'linear'
            else _np.log10(partial_power_matrix)
        )

        ax.imshow(
            partial_power_matrix,
            extent=[
                _np.min(periods),
                _np.max(periods),
                _np.min(lengths),
                _np.max(lengths),
            ],
            aspect='auto',
            origin='lower',
            norm=colors.Normalize(vmin=vmin, vmax=vmax)
            if cscale == 'linear'
            else colors.LogNorm(vmin=vmin, vmax=vmax),
        )

        sm = _plt.cm.ScalarMappable(_plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array(partial_power_matrix)

        cbar = fig.colorbar(
            sm,
            ax=ax,
            label='Partial Power [kW]',
            format='%.3f' if cscale == 'linear' else '%.0i',
        )
        cbar.set_ticks(_np.linspace(vmin, vmax, step))
        fig.tight_layout()
        if savefig:
            _plt.savefig(figname, dpi=dpi)
        else:
            _plt.show()

    def get_undulator_from_matrix(
        self, target_period: float, target_length: float, data: tuple
    ):
        """Get information about the target point in matrix.

        Args:
            target_period (float): Undulator period [mm]
            target_length (float): Undulator length [m]
            data (tuple): data especified to use in calculation
                First position 'flux matrix' or 'flux density matrix' or 'brilliance matrix'
                Second position unds matrix
        """  # noqa: E501, D202

        if data is None:
            raise ValueError("'data' parameter has to be defined")

        result_matrix = data[0]
        info_unds_matrix = data[1]
        und: Undulator = data[2]

        pts_period = len(result_matrix[0, :])
        pts_length = len(result_matrix[:, 0])

        max_period = _np.max(info_unds_matrix[:, 1])
        min_period = _np.min(info_unds_matrix[:, 1])

        max_length = _np.max(info_unds_matrix[:, 2])
        min_length = _np.min(info_unds_matrix[:, 2])

        rtol_length = 0.6 * (max_length - min_length) / pts_length
        rtol_period = 0.6 * (max_period - min_period) / pts_period

        idcs_period = _np.isclose(
            info_unds_matrix[:, 1], target_period, atol=rtol_period
        )
        idcs_p = _np.where(idcs_period)[0]

        idcs_length = _np.isclose(
            info_unds_matrix[idcs_p, 2], target_length, atol=rtol_length
        )
        idcs_l = _np.where(idcs_length)[0]

        idxs = idcs_p[idcs_l]

        print(
            '{:}{:<2}{:}{:<5}{:}{:<7}{:}{:<7}{:}{:<3}{:}{:<2}{:}{:<2}{:}{:<2}{:}'.format(
                'Und',
                '',
                'Keff',
                '',
                'Ky',
                '',
                'Kx',
                '',
                'Gap',
                '',
                'Period',
                '',
                'Length',
                '',
                'H Number',
                '',
                'Result',
            )
        )

        for i, idx in enumerate(idxs):
            k = info_unds_matrix[idx][0]
            kx = 0
            ky = 0
            gap = und.undulator_k_to_gap(
                k=k,
                period=info_unds_matrix[idx][1],
                br=und.br,
                a=und.halbach_coef[und.polarization]['a'],
                b=und.halbach_coef[und.polarization]['b'],
                c=und.halbach_coef[und.polarization]['c'],
            )
            if und.polarization == 'hp':
                ky = k
            elif und.polarization == 'vp':
                kx = k
            elif und.polarization == 'cp':
                kx = k / _np.sqrt(1 + und.fields_ratio**2)
                ky = kx * und.fields_ratio
            print(
                '{:}{:<4}{:.5f}{:<2}{:.5f}{:<2}{:.5f}{:<2}{:.2f}{:<2}{:.2f}{:<3}{:.2f}{:<4}{:}{:<9}{:.2e}'.format(
                    i,
                    '',
                    k,
                    '',
                    ky,
                    '',
                    kx,
                    '',
                    gap,
                    '',
                    info_unds_matrix[idx][1],
                    '',
                    info_unds_matrix[idx][2],
                    '',
                    int(info_unds_matrix[idx][3]),
                    '',
                    result_matrix.ravel()[idx],
                )
            )
