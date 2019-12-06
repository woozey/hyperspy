import pytest
import requests
from hyperspy.misc.eels.eelsdb import eelsdb
from requests.exceptions import SSLError
import warnings

def eelsdb_down():
    try:
        request = requests.get('http://api.eelsdb.eu', verify=False)
        return False
    except requests.exceptions.ConnectionError:
        return True


@pytest.mark.skipif(eelsdb_down(), reason="Unable to connect to EELSdb")
def test_eelsdb_eels():
    try:
        ss = eelsdb(
            title="Boron Nitride Multiwall Nanotube",
            formula="BN",
            spectrum_type="coreloss",
            edge="K",
            min_energy=370,
            max_energy=1000,
            min_energy_compare="gt",
            max_energy_compare="lt",
            resolution="0.7 eV",
            resolution_compare="lt",
            max_n=2,
            order="spectrumMin",
            order_direction='DESC',
            monochromated=False, )
    except SSLError:
        warnings.warn(
            "The https://eelsdb.eu certificate seems to be invalid. "
            "Consider notifying the issue to the EELSdb webmaster.")
        ss = eelsdb(
            title="Boron Nitride Multiwall Nanotube",
            formula="BN",
            spectrum_type="coreloss",
            edge="K",
            min_energy=370,
            max_energy=1000,
            min_energy_compare="gt",
            max_energy_compare="lt",
            resolution="0.7 eV",
            resolution_compare="lt",
            max_n=2,
            order="spectrumMin",
            order_direction='DESC',
            monochromated=False,
            verify_certificate=False)
    assert len(ss) == 2
    md = ss[0].metadata
    assert md.General.author == "Odile Stephan"
    assert (
        md.Acquisition_instrument.TEM.Detector.EELS.collection_angle == 24)
    assert md.Acquisition_instrument.TEM.convergence_angle == 15
    assert md.Acquisition_instrument.TEM.beam_energy == 100
    assert md.Signal.signal_type == "EELS"
    assert "perpendicular" in md.Sample.description
    assert "parallel" in ss[1].metadata.Sample.description
    assert md.Sample.chemical_formula == "BN"
    assert md.Acquisition_instrument.TEM.microscope == "STEM-VG"


@pytest.mark.skipif(eelsdb_down(), reason="Unable to connect to EELSdb")
def test_eelsdb_xas():
    try:
        ss = eelsdb(
            spectrum_type="xrayabs", max_n=1,)
    except SSLError:
        ss = eelsdb(
            spectrum_type="xrayabs", max_n=1, verify_certificate=False)
    assert len(ss) == 1
    md = ss[0].metadata
    assert md.Signal.signal_type == "XAS"
