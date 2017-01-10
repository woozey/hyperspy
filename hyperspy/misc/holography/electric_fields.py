# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
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
from hyperspy.misc.holography.holo_constants import c_e
from hyperspy.misc.utils import laplacian2d_complex
import scipy.constants as constants


def charge_density_map(wave, beam_energy):
    """
    Calculates charge density map from reconstructed wave using Laplacian operator

    Parameters
    ----------
    wave : ndarray of complex
        Electron wave image
    beam_energy : float
        Electron beam energy in keV

    Returns
    -------
    Interaction constant in SI units as a float
    """

    (gradient_x, gradient_y) = np.gradient(wave)
    charge = - (constants.epsilon_0 / c_e(beam_energy) / constants.e) * \
             np.imag(laplacian2d_complex(wave) / wave - (gradient_x ** 2 + gradient_y ** 2) / wave ** 2)

    return charge
