from rokpy.utilities import biot_characteristic_frequency
from rokpy.fluid_properties import BatzleWang

salinity = 120000
T = 70
P = 60
viscosity = BatzleWang.brine_viscosity(T, salinity)
density = BatzleWang.brine_density(T, P, salinity)

print([viscosity, density])