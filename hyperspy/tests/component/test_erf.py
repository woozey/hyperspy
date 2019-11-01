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
from numpy.testing import assert_allclose
import pytest
from distutils.version import LooseVersion
import sympy

from hyperspy.components1d import Erf

pytestmark = pytest.mark.skipif(LooseVersion(sympy.__version__) <
                                LooseVersion("1.3"),
                                reason="This test requires SymPy >= 1.3")

def test_function():
    g = Erf()
    g.A.value = 1
    g.sigma.value = 2
    g.origin.value = 3
    assert g.function(3) == 0.
    assert_allclose(g.function(15),0.5)
    assert_allclose(g.function(1.951198),-0.2,rtol=1e-6)

