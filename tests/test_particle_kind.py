from __future__ import annotations

import pytest

from mckit_meshes.particle_kind import ParticleKind


def test_creation():
    n = ParticleKind.n
    assert n.value == 1
    assert n.name == "neutron"
    assert n.short == "n"
    assert ParticleKind.n is ParticleKind.neutron


def test_long_name():
    e = ParticleKind.e
    assert e.name == "electron"


def test_creation_from_long_name():
    p = ParticleKind["photon"]
    assert p is ParticleKind.p


def test_heating_spec():
    spec = ParticleKind.p.heating_reactions
    assert spec == "-5 -6"
    spec = ParticleKind.n.heating_reactions
    assert spec == "1 -4"


def test_heating_spec_fail():
    e = ParticleKind.e
    with pytest.raises(ValueError, match="Heating spec is not defined for electron"):
        e.heating_reactions  # noqa: B018
