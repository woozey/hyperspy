# -*- coding: utf-8 -*-
#  Copyright 2007-2017 The HyperSpy developers
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

import numpy as np
import numpy.testing as nt
import hyperspy.api as hs


class TestCaseElectronWaveImage(object):

    def setUp(self):
        self.phase = 1
        self.wave_image = hs.signals.ElectronWaveImage(np.exp(1j * self.phase))

    def test_set_microscope_parameters(self):
        self.wave_image.set_microscope_parameters(beam_energy=300., biprism_voltage=80.5, tilt_alpha=2.2,
                                                  tilt_beta=0.)
        nt.assert_equal(self.holo_image.metadata.Acquisition_instrument.TEM.beam_energy, 300.)
        nt.assert_equal(self.holo_image.metadata.Acquisition_instrument.TEM.Holography.Biprism_voltage, 80.5)
        nt.assert_equal(self.holo_image.metadata.Acquisition_instrument.TEM.Tilt_alpha, 2.2)
        nt.assert_equal(self.holo_image.metadata.Acquisition_instrument.TEM.Tilt_beta, 0.)

    def test_charge_density_map(self):

        # 1. Testing raises:
        nt.assert_raises(AttributeError, self.wave_image.charge_density_map())

        self.wave_image.set_microscope_parameters(beam_energy=300.)

        wave_image_1 = self.wave_image.deepcopy()
        wave_image_1.axes_manager.signal_axes[0].scale = 0.1
        wave_image_1.axes_manager.signal_axes[1].scale = 0.1
        nt.assert_raises(AttributeError, wave_image_1.charge_density_map())

        wave_image_1.axes_manager.signal_axes[0].units = 'nm'
        nt.assert_raises(AttributeError, wave_image_1.charge_density_map())

        wave_image_1.axes_manager.signal_axes[0].units = 'nm'

        # 2. Testing charge density calculation without scale given:

if __name__ == '__main__':
    import nose
    nose.run(defaultTest=__name__)
