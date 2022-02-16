Dose evaluation for the DNFM case in IBR2 experiment
c
c
c
c  -------------------------------------  Cells
c  The steel case
1  1  -7.93  1  -4  5  -6   imp:p=1  $ cylinder wall
2  1  -7.93  1  -2 -5       imp:p=1  $ left  disc
3  1  -7.93  3  -4 -5       imp:p=1  $ right disc
c  Vacuum space
4  0         2  -3 -5  imp:p=1  $ inner volume
5  0         -1 -6 -7  imp:p=1
6  0         4  -6 -7  imp:p=1
7  0         6  -7     imp:p=1
c  Sceen boundary
8  0         7         imp:p=0

c  --------------------------------------  Surfaces
1  px   -20.0
2  px   -19.8
3  px    19.8
4  px    20.0
5  cx    3.8
6  cx    4.0
7  so    1000.0

c  -------------------------------------- Controls
m1
        6000.84p -0.0012
        13000.84p -0.001
        14000.84p -0.008
        15000.84p -0.0004
        16000.84p -0.0002
        22000.84p -0.008
        24000.84p -0.19
        25000.84p -0.02
        26000.84p -0.6512
        27000.84p -0.005
        28000.84p -0.11
        41000.84p -0.001
        42000.84p -0.002
        74000.84p -0.002
c
mode p
c
sdef par=2 axs=1 0 0 pos=0 0 0 ext=d1 rad=fext d2  erg=d5
sc1 Distribution along cylinder [-20...20]
si1 H -20 -19.8 19.8 20
sp1 D 0  10  194  10
ds2 Q -19.8 3 19.8 4 20 3
sc3 Distribution over radius in ranges [0..4.0] on discs
si3 0 4
si4 3.8 4.0
sc5   Time: 3 d
si5   H  0.0  0.01  0.02  0.05  0.10
            0.20  0.30  0.40  0.60
            0.80  1.00  1.22  1.44
            1.66  2.00  2.50  3.00
sp5   D  0.0  1.91157e+03  4.51474e+01  1.01153e+01  2.82464e+03
        1.08524e+03  2.69399e+01  1.58789e+02  2.54384e+03
        3.43669e+03  4.22058e+02  2.30433e+02  2.12135e+02
        3.29538e-02  1.81890e+00  2.14673e-03  3.39805e-06
c
rand gen=2 seed=501932527452161 stride=152917 hist=1
nps 1e+10
c ctme  5906
ctme  10
prdmp 1e9 1e9  1 1 1e9  $ ouput tallies, dump and tfc on every billion nps
print
c
fc1 === Total flux over the outer surface (to check normalization)
c Normalizing coefficient: steel mass * total gamma output = 2.19261e+07
fm1:p  2.19261e+07
f1:p  5
e1:p  3.0
c
fc5 === Gamma dose at 10 cm distance from cylinder surface, Sv/h
fm5:p  0.0789341
f5:p 0 0 14.0  1
c Dose conversion by "Recomendations..." IDM#29PJCT, pSv/(gamma/cm2)
de5:p
        0.01 0.015 0.02 0.03 0.04
        0.05 0.06 0.07 0.08 0.1
        0.15 0.2 0.3 0.4 0.5
        0.6 0.8 1.0 2.0 4.0
        6.0 8.0 10.0
df5:p
        0.0485 0.1254 0.205 0.2999 0.3381
        0.3572 0.378 0.4066 0.4399 0.5172
        0.7523 1.0041 1.5083 1.9958 2.4657
        2.9082 3.7269 4.4834 7.4896 12.0153
        15.9873 19.9191 23.76
e5:p  18.0
c
fc15 === Gamma dose at 1 m distance from cylinder center, Sv/h
fm15:p  0.0789341
f15:p 0 0 100.0  1
c Dose conversion by "Recomendations..." IDM#29PJCT, pSv/(gamma/cm2)
de15:p
      0.01 0.015 0.02 0.03 0.04
      0.05 0.06 0.07 0.08 0.1
      0.15 0.2 0.3 0.4 0.5
      0.6 0.8 1.0 2.0 4.0
      6.0 8.0 10.0
df15:p
      0.0485 0.1254 0.205 0.2999 0.3381
      0.3572 0.378 0.4066 0.4399 0.5172
      0.7523 1.0041 1.5083 1.9958 2.4657
      2.9082 3.7269 4.4834 7.4896 12.0153
      15.9873 19.9191 23.76
e15:p  18.0
c
fc14  === Gamma dose in mesh, Sv/h
fmesh14:p
      geom cyl
      origin -30 0 0
      axs 1 0 0
      vec 0 0 1
      imesh 3.8 4 20  iints 4 4 16
      jmesh 60        jints 60
      kmesh 1
      factor 0.0789341
de14:p
      0.01 0.015 0.02 0.03 0.04
      0.05 0.06 0.07 0.08 0.1
      0.15 0.2 0.3 0.4 0.5
      0.6 0.8 1.0 2.0 4.0
      6.0 8.0 10.0
df14:p
      0.0485 0.1254 0.205 0.2999 0.3381
      0.3572 0.378 0.4066 0.4399 0.5172
      0.7523 1.0041 1.5083 1.9958 2.4657
      2.9082 3.7269 4.4834 7.4896 12.0153
      15.9873 19.9191 23.76


