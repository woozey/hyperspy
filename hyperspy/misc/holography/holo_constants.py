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
import scipy.constants as constants


def c_e(beam_energy):
    """
    Calculates interaction constant C_E

    Parameters
    ----------
    beam_energy : float
        Electron beam energy in keV

    Returns
    -------
    Interaction constant in SI units as a float
    """

    beam_energy *= 1e3  # beam energy in eV

    r_beam_energy = beam_energy * (1 + constants.e * beam_energy / (2 * constants.electron_mass * constants.c ** 2))
    electron_me = constants.electron_mass * constants.c ** 2

    return 2 * np.pi * np.sqrt(2 * constants.e * constants.electron_mass * r_beam_energy) / \
           (beam_energy * constants.h) * (electron_me + constants.e * beam_energy) / \
           (2 * electron_me + constants.e * beam_energy)
