"""Particle kinds enumeration and methods."""


from enum import IntEnum


class ParticleKind(IntEnum):
    """Particle kinds enumeration and methods."""

    neutron = 1
    photon = 2
    electron = 3
    n = 1
    p = 2
    e = 3

    @property
    def short(self) -> str:
        """One letter synonym for a long name.

        Returns:
            the fires letter of the name
        """
        return self.name[0]

    @property
    def heating_reactions(self) -> str:
        """MCNP specified heating reactions for neutrons and photons.

        Returns:
            reaction specification for neutron or photon particle

        Raises:
            ValueError: if there's no specification for the particle
        """
        if self is self.n:
            return "1 -4"
        elif self is self.p:
            return "-5 -6"
        else:
            raise ValueError("Heating spec is not defined for " + self.name)
