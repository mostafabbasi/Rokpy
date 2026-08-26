from rokpy.constants import PropertyTemplates, MineralsPropertyTable, FluidsPropertyTable
from rokpy.materials import Mineral, Fluid, MineralSet, FluidSet, InclusionRock
from rokpy.effective_medium import BoundMethods, InclusionMethods, Inclusion, ShapeName
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from rokpy.visualization import Sheet, RockOptimizer
from rokpy.constants import PropertyTemplates
import lasio

from pathlib import Path
lasfile = r"C:\\Users\\MAbbasi\\MEGA\\Python\\Projects\\rokpy\\examples\\wellA.las"
lasfile = str(Path(__file__).resolve().parent) + '\wellA.las'
las = lasio.read(lasfile)


# Extract required well logs from the file
vp_log = las.curvesdict['VP']
vs_log = las.curvesdict['VS']
rho_log = las.curvesdict['RHOB']
phit_log = las.curvesdict['PHIT']
vcly_log = las.curvesdict['VCL']
sw_log = las.curvesdict['SW']

# Some of attributes of a curve:
print(f'Number of samples:  {vp_log.data.size}')
print(f'Log name:           {vp_log.mnemonic}')
print(f'Log unit:           {vp_log.unit}')
# Define PropertyTemplates
templates = PropertyTemplates()
# templates.PVelocity.plot_range=(1500., 3000.)
# templates.SVelocity.plot_range=(600., 1500.)
# templates.Density.plot_range=(1.45, 2.9)

# Define Log Sheet
sheet = plt.figure(FigureClass=Sheet, figsize=(12, 10))
sheet.set_depth_range(1950,2050)

# Add tracks to the sheet
vol_tr = sheet.add_track(templates.VolumeFraction)
por_tr = sheet.add_track(templates.Porosity)
sat_tr = sheet.add_track(templates.Saturation)
vp_tr = sheet.add_track(templates.PVelocity)
vs_tr = sheet.add_track(templates.SVelocity)
ro_tr = sheet.add_track(templates.Density)
opt_tr = sheet.add_empty_axes()

# Disply each logs on the corresponding tracks
por_tr.plot(las.depth_m, phit_log.data, label=phit_log.mnemonic)
vp_tr.plot(las.depth_m, vp_log.data, label=vp_log.mnemonic)
vs_tr.plot(las.depth_m, vs_log.data, label=vs_log.mnemonic)
ro_tr.plot(las.depth_m, rho_log.data, label=rho_log.mnemonic)

# Define, modify and display the mineralset
minerals = MineralsPropertyTable()
minerals.DryClay.set_properties(27, 7, 2.9)
minerals.Quartz.set_properties(45, 45, 2.65)
clay = Mineral(minerals.DryClay)
quartz = Mineral(minerals.Quartz)
mineralset = MineralSet({clay: vcly_log.data, quartz:1-vcly_log.data}, BoundMethods.MixingMethodName.VoigtReussHill)
vol_tr.plot_component_set(las.depth_m, mineralset)

# Define, modify and display the fluidset
fluid_props = FluidsPropertyTable()
fluid_props.Brine.set_properties(2.37, 0, 1.1)
fluid_props.Oil.set_properties(1.25, 0, 0.45)
brine = Fluid(fluid_props.Brine)
oil = Fluid(fluid_props.Oil)
fluidset = FluidSet({brine: sw_log.data, oil:1-sw_log.data})
sat_tr.plot_component_set(las.depth_m, fluidset)


# Define Rock
phit = phit_log.data
rock = InclusionRock( mineralset, fluidset, phit, InclusionMethods.InclusionMethodName.DEM)
# clay_inclusion = Inclusion(ShapeName.SPHEROID, 0.05, fluidset.bound_fluid, clay)
clay_inclusion = Inclusion(ShapeName.SPHEROID, 0.07)
quartz_inclusion = Inclusion(ShapeName.SPHEROID, 0.11)

# rock.add_inclusion(clay_inclusion, clay)
rock.add_inclusion(quartz_inclusion, quartz)
quartz.porosity_weight = 3

optimizer = RockOptimizer(opt_tr, sheet, rock, las.depth_m)
plt.show()
