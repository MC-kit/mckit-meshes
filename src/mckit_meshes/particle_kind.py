from enum import IntEnum


class ParticleKind(IntEnum):
    neutron = (1,)
    photon = (2,)
    electron = (3,)
    n = (1,)
    p = (2,)
    e = 3

    @property
    def short(self):
        return self.name[0]

    @property
    def heating_reactions(self):
        if self is self.n:
            return "1 -4"
        elif self is self.p:
            return "-5 -6"
        else:
            raise RuntimeError("Heating spec is not defined for " + self.name)
