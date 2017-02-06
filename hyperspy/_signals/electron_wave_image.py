# -*- coding: utf-8 -*-
# Copyright 2007-2017 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.


from hyperspy._signals.complex_signal2d import ComplexSignal2D
from hyperspy.misc.holography.electric_fields import charge_density_map
from traits.trait_base import _Undefined


class ElectronWaveImage(ComplexSignal2D):

    """ComplexSignal2D subclass for electron wave images."""

    _signal_type = 'electron_wave'

    # The class is empty at the moment, but some electron wave specific methods will be added later.

    # @property
    # def reconstruction_parameters(self):
    #     assert self.metadata.Signal.has_item('holo_reconstruction_parameters'), \
    #         "No reconstruction parameters assigned to the wave"
    #
    #     return self.metadata.Signal.holo_reconstruction_parameters.as_dictionary()

    def set_microscope_parameters(self,
                                  beam_energy=None,
                                  biprism_voltage=None,
                                  tilt_stage=None):
        """Set the microscope parameters.

        If no arguments are given, raises an interactive mode to fill
        the values.

        Parameters
        ----------
        beam_energy: float
            The energy of the electron beam in keV
        biprism_voltage : float
            In volts
        tilt_stage : float
            In degrees

        Examples
        --------

        >>> s.set_microscope_parameters(beam_energy=300.)
        >>> print('Now set to %s keV' %
        >>>       s.metadata.Acquisition_instrument.
        >>>       TEM.beam_energy)

        Now set to 300.0 keV

        """
        md = self.metadata

        if beam_energy is not None:
            md.set_item("Acquisition_instrument.TEM.beam_energy", beam_energy)
        if biprism_voltage is not None:
            md.set_item(
                "Acquisition_instrument.TEM.Biprism.voltage",
                biprism_voltage)
        if tilt_stage is not None:
            md.set_item("Acquisition_instrument.TEM.tilt_stage", tilt_stage)

    def charge_density_map(self, show_progressbar=False):
        """
        Calculates charge density map using numerical Laplacian as described in [Beleggia, M. et al.
        Direct measurement of the charge distribution along a biased carbon nanotube bundle using electron holography.
        Applied Physics Letters 98, 243101 (2011). http://dx.doi.org/10.1063/1.3598468].

        Parameters
        ----------
        show_progressbar : boolean
            Shows progressbar while iterating over different slices of the signal (passes the parameter to map method).

        Returns
        -------
        Signal2D instance of charge density map in e/(original axes units)^2 or e/px if original units are not given.
        For example if signal axes of original wave had nm units, the output will be in e/nm^2
        """

        try:
            ht = self.metadata.Acquisition_instrument.TEM.beam_energy
        except:
            raise AttributeError("Please define the beam energy."
                                 "You can do this e.g. by using the "
                                 "set_microscope_parameters method")

        # Dealing with units and pixel size:
        if isinstance(self.axes_manager.signal_axes[0].units, _Undefined) or \
                isinstance(self.axes_manager.signal_axes[1].units, _Undefined):
            if self.axes_manager.signal_axes[0].scale != 1 or self.axes_manager.signal_axes[0].scale != 1:
                raise AttributeError('The units of signal axes are not set though the scale is not one, which is '
                                     'confusing. Please fix it and re-execute the code.')
            new_units = "e/px"
        elif self.axes_manager.signal_axes[0].units != self.axes_manager.signal_axes[1].units:
            raise AttributeError('The units of two signal axes are not equal. Please convert the units and re-execute '
                                 'the code.')
        else:
            new_units = "e/" + self.axes_manager.signal_axes[0].units + "^2"

        charge_density = self.map(charge_density_map, beam_energy=ht, inplace=False, show_progressbar=show_progressbar)
        charge_density.set_signal_type('signal2d')

        pixel_size = self.axes_manager.signal_axes[0].scale * self.axes_manager.signal_axes[1].scale

        charge_density /= pixel_size
        new_quantity = "Charge density (" + new_units + ")"
        charge_density.metadata.set_item("Signal.quantity", new_quantity)

        return charge_density
