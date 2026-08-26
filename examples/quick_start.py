import rokpy as rp

shale = rp.Material(p_velocity=3200, s_velocity=1800, density=2.4)

print(shale.poisson)