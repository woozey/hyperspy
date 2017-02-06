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
import scipy.constants as constants
from scipy.ndimage.filters import median_filter, gaussian_filter
import hyperspy.api as hs
from hyperspy.misc.holography.holo_constants import c_e


class TestCaseElectronWaveImage(object):

    def setUp(self):
        self.beam_energy = 300.
        self.img_sizex = 128
        self.img_sizey = 64
        c = 1e-6
        y, x = np.meshgrid(np.arange(self.img_sizey), np.arange(self.img_sizex))
        charge_coordinates_x = np.round(np.random.rand(6) * 63)
        charge_coordinates_y = np.round(np.random.rand(6) * 63)

        # coordinates of mirror plane:

        a = self.img_sizex/2
        b = 0

        charge_mirror_x = (b ** 2 - a ** 2) / (a ** 2 + b ** 2) * charge_coordinates_x -\
                          2 * a * b / (a ** 2 + b ** 2) * charge_coordinates_y + 2 * a
        charge_mirror_y = (a ** 2 - b ** 2) / (a ** 2 + b ** 2) * charge_coordinates_y - \
                          2 * a * b / (a ** 2 + b ** 2) * charge_coordinates_x + 2 * b
        charge = np.int64(np.random.rand(6) * 10)

        self.phase = np.zeros((6, self.img_sizex, self.img_sizey))
        self.input_charge = np.zeros((6, self.img_sizex, self.img_sizey))

        for i in range(6):
            self.phase[i, :, :] = - charge[i] * c_e(self.beam_energy) * constants.e / np.pi / 4 / constants.epsilon_0 \
                                  * np.log(np.sqrt((x - charge_coordinates_x[i] + c) ** 2 +
                                                   (y - charge_coordinates_y[i] + c) ** 2) /
                                           np.sqrt(
                                               (x - charge_mirror_x[i] + c) ** 2 + (y - charge_mirror_y[i] + c) ** 2))

            self.input_charge[i, int(charge_coordinates_x[i]), int(charge_coordinates_y[i])] = charge[i]
            self.input_charge[i, int(charge_mirror_x[i]), int(charge_mirror_y[i])] = - charge[i]

        self.wave_image = hs.signals.ElectronWaveImage(np.reshape(np.exp(1j * self.phase),
                                                                  (2, 3, self.img_sizex, self.img_sizey)))

    def test_set_microscope_parameters(self):
        self.wave_image.set_microscope_parameters(beam_energy=self.beam_energy, biprism_voltage=80.5, tilt_stage=2.2)
        nt.assert_equal(self.wave_image.metadata.Acquisition_instrument.TEM.beam_energy, self.beam_energy)
        nt.assert_equal(self.wave_image.metadata.Acquisition_instrument.TEM.Biprism.voltage, 80.5)
        nt.assert_equal(self.wave_image.metadata.Acquisition_instrument.TEM.tilt_stage, 2.2)

    def test_charge_density_map(self):

        # 1. Testing raises:
        nt.assert_raises(AttributeError, self.wave_image.charge_density_map)

        self.wave_image.set_microscope_parameters(beam_energy=self.beam_energy)

        wave_image_1 = self.wave_image.deepcopy()
        wave_image_1.axes_manager.signal_axes[0].scale = 0.1
        wave_image_1.axes_manager.signal_axes[1].scale = 0.1
        nt.assert_raises(AttributeError, wave_image_1.charge_density_map)

        wave_image_1.axes_manager.signal_axes[0].units = 'nm'
        nt.assert_raises(AttributeError, wave_image_1.charge_density_map)

        wave_image_1.axes_manager.signal_axes[1].units = 'nm'

        # 2. Testing charge density calculation without scale given:

        charge_density = self.wave_image.charge_density_map()
        charge_density_median = charge_density.map(median_filter, size=3, inplace=False)
        charge_density_gauss = charge_density.map(gaussian_filter, sigma=0.61, inplace=False)
        charge_density_1 = wave_image_1.charge_density_map()

if __name__ == '__main__':
    import nose
    nose.run(defaultTest=__name__)
