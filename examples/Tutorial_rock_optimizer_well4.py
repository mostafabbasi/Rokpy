from rokpy.constants import PropertyTemplates, MineralsPropertyTable, FluidsPropertyTable
from rokpy.materials import Mineral, Fluid, MineralSet, FluidSet, InclusionRock
from rokpy.effective_medium import BoundMethods, InclusionMethods, Inclusion, ShapeName
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from rokpy.visualization import Sheet, RockOptimizer
from rokpy.constants import PropertyTemplates
import lasio
from rokpy.visualization import Sheet
from rokpy.utilities import moving_median
from rokpy.materials import Mineral, Fluid, MineralSet, FluidSet, InclusionRock
from rokpy.effective_medium import Inclusion
from rokpy.constants import MineralsPropertyTable, FluidsPropertyTable
from rokpy.conversions import psi_to_mpa
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
file = r"C:\\Users\\MAbbasi\\MEGA\\Python\\Projects\\rokpy\\examples\\well4.las"

las = lasio.read(file)
print(las.keys())

# Read logs from las for more readability
depth_range = [2650, 2950]
mask = (las.depth_m>depth_range[0]) & (las.depth_m<depth_range[1])
depth = las.depth_m[mask]

vclay   = moving_median(las.curvesdict['VCLAY'].data[mask])
vsh     = moving_median(las.curvesdict['VSH'].data[mask])
vqz     = moving_median(las.curvesdict['VQZ'].data[mask])
vcal    = moving_median(las.curvesdict['VCAL'].data[mask])
phit    = moving_median(las.curvesdict['PHIT'].data[mask])
phibw   = moving_median(las.curvesdict['PHI_BOUND'].data[mask])
phiiso  = moving_median(las.curvesdict['PHI_ISOLATED'].data[mask])
swt     = moving_median(las.curvesdict['SWT'].data[mask])
sot     = moving_median(las.curvesdict['SOT'].data[mask])
sgt     = moving_median(las.curvesdict['SGT'].data[mask])
rho     = moving_median(las.curvesdict['RHOB'].data[mask])
vp      = moving_median(las.curvesdict['VP'].data[mask])
vs      = moving_median(las.curvesdict['VS'].data[mask])

minerals = MineralsPropertyTable()
fluids = FluidsPropertyTable()
fluids.calculate_brine(T=40, P=psi_to_mpa(2984.), salinity=50000)
print(fluids.Brine)

clay = Mineral(minerals.DryClay)
quartz = Mineral(minerals.Quartz)
calcite = Mineral(minerals.Calcite)

brine = Fluid(fluids.Brine)
oil = Fluid(fluids.Oil)
gas = Fluid(fluids.Gas)

mineralset = MineralSet({clay:vclay, calcite:vcal, quartz:vqz}, mixing_method='voigt_reuss_hill')
fluidset = FluidSet({brine:swt, oil:sot, gas:sgt})

rock = InclusionRock(mineralset, fluidset, phit)
rock.add_inclusion(Inclusion('SPHEROID',0.12), clay)
rock.add_inclusion(Inclusion('SPHEROID',0.09), calcite)
rock.add_inclusion(Inclusion('SPHEROID',0.2), quartz)




import matplotlib.pyplot as plt
from rokpy.constants import PropertyTemplates
sheet = plt.figure(FigureClass=Sheet, figsize=(20,9))
sheet.set_depth_range(*depth_range)

templates = PropertyTemplates()
templates.PVelocity.plot_range = (1000,6000)
templates.SVelocity.plot_range = (500,3000)
templates.Density.plot_range = (1.5,3)

por_tr = sheet.add_track(templates.Porosity)
vol_tr = sheet.add_track(templates.VolumeFraction)
sat_tr = sheet.add_track(templates.Saturation)
vp_tr = sheet.add_track(templates.PVelocity)
vs_tr = sheet.add_track(templates.SVelocity)
rho_tr = sheet.add_track(templates.Density)


porosity_fractionset = rock.minerals_porosity_set()
por_tr.plot(depth, phit, label='phit')
por_tr.plot_fraction_set(depth, porosity_fractionset)
por_tr.plot(depth, phibw, label='phi_bound', color='b',linewidth=1, linestyle=":")
vol_tr.plot_component_set(depth, mineralset)
sat_tr.plot_component_set(depth, fluidset)
vp_tr.plot(depth, vp)
vs_tr.plot(depth, vs)
rho_tr.plot(depth, rho)


# Tune the rock model
minerals.DryClay.set_properties(60, 25, 3.05)
minerals.Calcite.set_properties(68, 31, 2.71)
minerals.Quartz.set_properties(36.6, 42, 2.65)
print(mineralset.bulk)
fluids.Oil.density, fluids.Oil.bulk = 0.9, 0.7

rock.inclusions[0].aspect_ratio = 0.27  #clay 
rock.inclusions[1].aspect_ratio = 0.06  #calcite
rock.inclusions[2].aspect_ratio = 0.08  #quartz
calcite.porosity_weight = 0.05


opt_tr = sheet.add_empty_axes()
optimizer = RockOptimizer(opt_tr, sheet, rock, depth)
plt.show()
