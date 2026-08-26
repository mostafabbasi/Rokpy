"""Relations to calculate fluid properties under in-situ conditions

Implementation of Batzle & Wang (1992) and modified batzle and Wang (Xu, 2006) empirical relations 
for estimating elastic properties (density, velocity, bulk modulus, viscosity) of reservoir
fluids (water, brine, oil (live and dead), gas and CO₂) under insitu conditions.
"""

from rokpy.conversions import api_to_rho, velocity_to_bulk
import numpy as np
import warnings


class BatzleWang:
    """
    Implementation of Batzle & Wang (1992) empirical relations for estimating
    physical properties (density, velocity, bulk modulus, viscosity) of reservoir
    fluids: water, brine, oil (live and dead), and gas. Includes extensions for CO₂
    based on Xu (2006).

    References
    ----------
    Batzle, M., & Wang, Z. (1992). Seismic properties of pore fluids. Geophysics, 57(11), 1396–1408.
    Xu, H. (2006). Calculation of CO2 acoustic properties using Batzle-Wang equations. Geophysics, 71(2), F21–F23.
    """

    @staticmethod
    def water_density(T, P):
        """
        Compute pure water density using Batzle & Wang (1992) equation.

        Parameters
        ----------
        T : array_like
            Temperature in degrees Celsius (°C).
        P : array_like
            Pressure in megapascals (MPa).

        Returns
        -------
        rho : ndarray
            Water density in g/cm³.
        """
        return 1 + 1e-6 * (-80*T - 3.3*T**2 + 0.00175*T**3 + 489*P - 2*T*P + 0.016*P*T**2 - 1.3e-5*P*T**3 - 0.333*P**2 - 0.002*T*P**2)

    @staticmethod
    def water_velocity(T, P):
        """
        Compute pure water P-wave velocity using Batzle & Wang (1992) polynomial.

        Parameters
        ----------
        T : array_like
            Temperature in degrees Celsius (°C).
        P : array_like
            Pressure in megapascals (MPa).

        Returns
        -------
        vp : ndarray
            P-wave velocity in m/s.

        Warns
        -----
        UserWarning
            If any pressure value exceeds 100 MPa, as accuracy degrades beyond this limit.
        """
        if np.any(P > 100):
            warnings.warn('P values above about 100 MPa may result in invalid estimations')
        w = np.array([[1.40285e+03, 1.52400e+00, 3.43700e-03, -1.19700e-05],
                      [4.87100e+00, -1.11000e-02, 1.73900e-04, -1.62800e-06],
                      [-4.78300e-02, 2.74700e-04, -2.13500e-06, 1.23700e-08],
                      [1.48700e-04, -6.50300e-07, -1.45500e-08, 1.32700e-10],
                      [-2.19700e-07, 7.98700e-10, 5.23000e-11, -4.61400e-13]])
        return sum(w[i, j] * T**i * P**j for i in range(5) for j in range(4))

    @staticmethod
    def brine_density(T, P, salinity):
        """
        Compute brine density from water density corrected for salinity.

        Parameters
        ----------
        T : array_like
            Temperature in °C.
        P : array_like
            Pressure in MPa.
        salinity : array_like
            Salinity in parts per million (ppm).

        Returns
        -------
        rho : ndarray
            Brine density in g/cm³.
        """
        salinity = salinity / 1e6
        water_density = BatzleWang.water_density(T, P)
        return water_density + salinity * (0.668 + 0.44 * salinity + 1e-6 * (300 * P - 2400 * P * salinity + T * (80 + 3 * T - 3300 * salinity - 13 * P + 47 * P * salinity)))

    @staticmethod
    def brine_velocity(T, P, salinity):
        """
        Compute brine P-wave velocity using Batzle & Wang (1992) relation.

        Parameters
        ----------
        T : array_like
            Temperature in °C.
        P : array_like
            Pressure in MPa.
        salinity : array_like
            Salinity in ppm.

        Returns
        -------
        vp : ndarray
            Brine P-wave velocity in m/s.
        """
        salinity = salinity / 1e6
        Vw = BatzleWang.water_velocity(T, P)
        return Vw + salinity * (1170 - 9.6 * T + 0.055 * T**2 - 8.5e-5 * T**3 + 2.6 * P - 0.0029 * T * P - 0.0476 * P**2) + (salinity**1.5) * (780 - 10 * P + 0.16 * P**2) - 820 * salinity**2

    @staticmethod
    def brine_viscosity(T, salinity):
        """
        Estimate brine viscosity using Batzle & Wang (1992) empirical relation.

        Parameters
        ----------
        T : array_like
            Temperature in °C.
        salinity : array_like
            Salinity in ppm.

        Returns
        -------
        mu : ndarray
            Brine viscosity in centipoise (cP).
        """
        salinity = salinity / 1e6
        return 0.1 + 0.333 * salinity + (1.65 + 91.9 * salinity**3) * np.exp(-(0.42 * (salinity**0.8 - 0.17)**2 + 0.045) * T**0.8)

    @staticmethod
    def bubble_oil_density(T, api, gor, gas_gravity):
        """
        Compute oil density at bubble-point pressure.

        Parameters
        ----------
        T : array_like
            Temperature in °C.
        api : array_like
            API gravity (degrees API).
        gor : array_like
            Gas-oil ratio in scf/STB.
        gas_gravity : array_like
            Gas specific gravity (relative to air).

        Returns
        -------
        rho_bubble : ndarray
            Oil density at bubble-point in g/cm³.
        """
        Bob = BatzleWang.bubble_formation_factor(T, api, gor, gas_gravity)
        density_standard = api_to_rho(api)
        return (density_standard + 0.0012 * gas_gravity * gor) / Bob

    @staticmethod
    def oil_density(T, P, api, gor, gas_gravity):
        """
        Compute live oil density at given pressure and temperature.

        Parameters
        ----------
        T : array_like
            Temperature in °C.
        P : array_like
            Pressure in MPa.
        api : array_like
            API gravity.
        gor : array_like
            Gas-oil ratio in scf/STB.
        gas_gravity : array_like
            Gas specific gravity.

        Returns
        -------
        rho : ndarray
            Live oil density in g/cm³.
        """
        density_bubble = BatzleWang.bubble_oil_density(T, api, gor, gas_gravity)
        density = BatzleWang.insitu_oil_density(T, P, density_bubble)
        return density

    @staticmethod
    def oil_velocity(T, P, api, gor, gas_gravity):
        """
        Compute live oil P-wave velocity using Batzle & Wang (1992).

        Parameters
        ----------
        T : array_like
            Temperature in °C.
        P : array_like
            Pressure in MPa.
        api : array_like
            API gravity.
        gor : array_like
            Gas-oil ratio in scf/STB.
        gas_gravity : array_like
            Gas specific gravity.

        Returns
        -------
        vp : ndarray
            Live oil P-wave velocity in m/s.
        """
        density_standard = api_to_rho(api)
        Bo = BatzleWang.bubble_formation_factor(T, api, gor, gas_gravity)
        pseudo_density = density_standard / Bo / (1 + 0.001 * gor)
        return 2096 * np.sqrt(pseudo_density / (2.6 - pseudo_density)) - 3.7 * T + 4.64 * P + 0.0115 * (4.12 * np.sqrt(1.08 / pseudo_density - 1) - 1) * T * P

    @staticmethod
    def oil_viscosity(T, P, api, gor, gas_gravity):
        """
        Estimate live oil viscosity using Batzle & Wang (1992) model.

        Parameters
        ----------
        T : array_like
            Temperature in °C.
        P : array_like
            Pressure in MPa.
        api : array_like
            API gravity.
        gor : array_like
            Gas-oil ratio in scf/STB.
        gas_gravity : array_like
            Gas specific gravity.

        Returns
        -------
        mu : ndarray
            Oil viscosity in centipoise (cP).
        """
        density_bubble = BatzleWang.bubble_oil_density(T, api, gor, gas_gravity)
        y = np.power(10, 5.693 - 2.863 / density_bubble)
        viscosity_standard = np.power(10, 0.505 * y / (17.8 + T)**1.163) - 1
        I = np.power(10, 18.6 * (0.1 * np.log10(viscosity_standard) + 1 / (np.log10(viscosity_standard) + 2)**0.1 - 0.985))
        return viscosity_standard + 0.145 * P * I

    @staticmethod
    def bubble_pressure(T, api, gor, gas_gravity):
        """
        Estimate bubble-point pressure using Batzle & Wang (1992).

        Parameters
        ----------
        T : array_like
            Temperature in °C.
        api : array_like
            API gravity.
        gor : array_like
            Gas-oil ratio in scf/STB.
        gas_gravity : array_like
            Gas specific gravity.

        Returns
        -------
        Pb : ndarray
            Bubble-point pressure in MPa.
        """
        return (gor / 2.03 / gas_gravity)**0.8299 * np.exp(-0.02878 * api + 0.00377 * T) - 0.176

    @staticmethod
    def bubble_formation_factor(T, api, gor, gas_gravity):
        """
        Compute oil formation volume factor at bubble-point.

        Parameters
        ----------
        T : array_like
            Temperature in °C.
        api : array_like
            API gravity.
        gor : array_like
            Gas-oil ratio in scf/STB.
        gas_gravity : array_like
            Gas specific gravity.

        Returns
        -------
        Bo : ndarray
            Formation volume factor (dimensionless, bbl/STB).
        """
        density_standard = api_to_rho(api)
        return 0.972 + 0.00038 * (2.495 * gor * np.sqrt(gas_gravity / density_standard) + T + 17.8)**1.175

    @staticmethod
    def maxgor(T, P, api, gas_gravity):
        """
        Estimate maximum gas-oil ratio (GOR) solubility at given P and T.

        Parameters
        ----------
        T : array_like
            Temperature in °C.
        P : array_like
            Pressure in MPa.
        api : array_like
            API gravity.
        gas_gravity : array_like
            Gas specific gravity.

        Returns
        -------
        gor_max : ndarray
            Maximum GOR in scf/STB.
        """
        density_standard = api_to_rho(api)
        return 0.02123 * gas_gravity * (P * np.exp(4.072 / density_standard - 0.00377 * T))**1.205

    @staticmethod
    def insitu_oil_density(T, P, standard_oil_density):
        """
        Compute in-situ oil density from standard (surface) density.

        Parameters
        ----------
        T : array_like
            Temperature in °C.
        P : array_like
            Pressure in MPa.
        standard_oil_density : array_like
            Standard oil density in g/cm³.

        Returns
        -------
        rho_insitu : ndarray
            In-situ oil density in g/cm³.
        """
        density_insituP = standard_oil_density + (0.00277 * P - 1.71e-7 * P**3) * (standard_oil_density - 1.15)**2 + 3.49e-4 * P
        return density_insituP / (0.972 + 3.81e-4 * (T + 17.78)**1.175)

    @staticmethod
    def gas_density(T, P, gas_gravity):
        """
        Compute natural gas density using real gas equation with Batzle-Wang Z-factor.

        Parameters
        ----------
        T : array_like
            Temperature in °C.
        P : array_like
            Pressure in MPa.
        gas_gravity : array_like
            Gas specific gravity.

        Returns
        -------
        rho : ndarray
            Gas density in g/cm³.
        """
        Ta = T + 273.15
        Pr = P / (4.892 - 0.4048 * gas_gravity)
        Tr = Ta / (94.72 + 170.75 * gas_gravity)
        R = 8.31441
        a = 0.03 + 0.00527 * (3.5 - Tr)**3
        b = 0.642 * Tr - 0.007 * Tr**4 - 0.52
        c = 0.109 * (3.85 - Tr)**2
        d = np.exp(-(0.45 + 8 * (0.56 - 1 / Tr)**2) * (Pr**1.2) / Tr)
        E = c * d
        Z = a * Pr + b + E
        return 28.8 * gas_gravity * P / (Z * R * Ta)

    @staticmethod
    def gas_velocity(T, P, gas_gravity):
        """
        Compute natural gas P-wave velocity from bulk modulus and density.

        Parameters
        ----------
        T : array_like
            Temperature in °C.
        P : array_like
            Pressure in MPa.
        gas_gravity : array_like
            Gas specific gravity.

        Returns
        -------
        vp : ndarray
            Gas P-wave velocity in m/s.
        """
        density_gas = BatzleWang.gas_density(T, P, gas_gravity)
        K_gas = BatzleWang.gas_bulk(T, P, gas_gravity)
        return 1000 * np.sqrt(K_gas / density_gas)

    @staticmethod
    def gas_bulk(T, P, gas_gravity):
        """
        Compute natural gas bulk modulus using Batzle & Wang (1992).

        Parameters
        ----------
        T : array_like
            Temperature in °C.
        P : array_like
            Pressure in MPa.
        gas_gravity : array_like
            Gas specific gravity.

        Returns
        -------
        K : ndarray
            Gas bulk modulus in GPa.
        """
        Ta = T + 273.15
        Pr = P / (4.892 - 0.4048 * gas_gravity)
        Tr = Ta / (94.72 + 170.75 * gas_gravity)
        a = 0.03 + 0.00527 * (3.5 - Tr)**3
        b = 0.642 * Tr - 0.007 * Tr**4 - 0.52
        c = 0.109 * (3.85 - Tr)**2
        d = np.exp(-(0.45 + 8 * (0.56 - 1 / Tr)**2) * (Pr**1.2) / Tr)
        E = c * d
        Z = a * Pr + b + E
        m = -1.2 * (Pr**0.2) * (0.45 + 8 * (0.56 - 1 / Tr)**2) / Tr
        f = c * d * m + a
        gamma = 0.85 + 5.6 / (Pr + 2) + 27.1 / (Pr + 3.5)**2 - 8.7 * np.exp(-0.65 * (Pr + 1))
        return P * gamma / (1 - Pr / Z * f) / 1000

    @staticmethod
    def gassybrine_bulk(Kb, gwr):
        """
        Adjust brine bulk modulus for gas-water ratio (GWR).

        Parameters
        ----------
        Kb : array_like
            Brine bulk modulus in GPa.
        gwr : array_like
            Gas-water ratio (dimensionless or consistent units).

        Returns
        -------
        K : ndarray
            Effective bulk modulus of gassy brine in GPa.
        """
        return Kb / (1 + 0.0494 * gwr)

    @staticmethod
    def deadoil_density(T, P, api):
        """
        Compute dead oil (no dissolved gas) density.

        Parameters
        ----------
        T : array_like
            Temperature in °C.
        P : array_like
            Pressure in MPa.
        api : array_like
            API gravity.

        Returns
        -------
        rho : ndarray
            Dead oil density in g/cm³.
        """
        density_standard = api_to_rho(api)
        density_insituP = density_standard + (0.00277 * P - 1.71e-7 * P**3) * (density_standard - 1.15)**2 + 3.49e-4 * P
        return density_insituP / (0.972 + 3.81e-4 * (T + 17.78)**1.175)

    @staticmethod
    def deadoil_velocity(T, P, api):
        """
        Compute dead oil P-wave velocity using Batzle & Wang (1992).

        Parameters
        ----------
        T : array_like
            Temperature in °C.
        P : array_like
            Pressure in MPa.
        api : array_like
            API gravity.

        Returns
        -------
        vp : ndarray
            Dead oil P-wave velocity in m/s.
        """
        density_standard = api_to_rho(api)
        return 2096 * np.sqrt(density_standard / (2.6 - density_standard)) - 3.7 * T + 4.64 * P + 0.0115 * (4.12 * np.sqrt(1.08 / density_standard - 1) - 1) * T * P

    @staticmethod
    def co2_bulk(P, T):
        """
        Compute CO₂ bulk modulus using modified Batzle-Wang equations (Xu, 2006).

        Parameters
        ----------
        P : array_like
            Pressure in MPa.
        T : array_like
            Temperature in °C.

        Returns
        -------
        K : ndarray
            CO₂ bulk modulus in GPa.

        References
        ----------
        Xu, H. (2006). Calculation of CO2 acoustic properties using Batzle-Wang equations.
        Geophysics, 71(2), F21–F23.
        """
        Ta = T + 273.15
        Pr = P / 7.4
        Tr = Ta / (31.1 + 273.5)
        a = 0.03 + 0.00527 * (3.5 - Tr)**3
        b = 0.642 * Tr - 0.007 * Tr**4 - 0.52
        c = 0.109 * (3.85 - Tr)**2
        d = np.exp(-(0.45 + 8 * (0.56 - 1 / Tr)**2) * (Pr**1.2) / Tr)
        E = c * d
        Z = a * Pr + b + E
        m = -1.2 * (Pr**0.2) * (0.45 + 8 * (0.56 - 1 / Tr)**2) / Tr
        f = c * d * m + a
        gamma = 0.85 + 5.6 / (Pr + 2) + 27.1 / (Pr + 3.5)**2 - 8.7 * np.exp(-0.65 * (Pr + 1))
        return P * gamma / (1 - Pr / Z * f) / 1000

    @staticmethod
    def co2_density(T, P, gas_gravity):
        """
        Compute CO₂ density using modified Batzle-Wang equations (Xu, 2006).

        Parameters
        ----------
        T : array_like
            Temperature in °C.
        P : array_like
            Pressure in MPa.
        gas_gravity : array_like
            Gas specific gravity (typically ~1.52 for pure CO₂).

        Returns
        -------
        rho : ndarray
            CO₂ density in g/cm³.

        References
        ----------
        Xu, H. (2006). Calculation of CO2 acoustic properties using Batzle-Wang equations.
        Geophysics, 71(2), F21–F23.
        """
        Ta = T + 273.15
        Pr = P / 7.4
        Tr = Ta / (31.1 + 273.5)
        R = 8.31441
        a = 0.03 + 0.00527 * (3.5 - Tr)**3
        b = 0.642 * Tr - 0.007 * Tr**4 - 0.52
        c = 0.109 * (3.85 - Tr)**2
        d = np.exp(-(0.45 + 8 * (0.56 - 1 / Tr)**2) * (Pr**1.2) / Tr)
        E = c * d
        Z = a * Pr + b + E
        return 28.8 * gas_gravity * P / (Z * R * Ta)

    @staticmethod
    def co2_velocity(T, P, gas_gravity):
        """
        Compute CO₂ P-wave velocity from bulk modulus and density.

        Parameters
        ----------
        T : array_like
            Temperature in °C.
        P : array_like
            Pressure in MPa.
        gas_gravity : array_like
            Gas specific gravity.

        Returns
        -------
        vp : ndarray
            CO₂ P-wave velocity in m/s.
        """
        density_gas = BatzleWang.co2_density(T, P, gas_gravity)
        K_gas = BatzleWang.co2_bulk(T, P)
        return 1000 * np.sqrt(K_gas / density_gas)


    @staticmethod
    def deadoil_viscosity(T, P, api):
        """
        Estimate dead oil viscosity (misnamed method; computes viscosity, not velocity).

        Parameters
        ----------
        T : array_like
            Temperature in °C.
        P : array_like
            Pressure in MPa.
        api : array_like
            API gravity.

        Returns
        -------
        mu : ndarray
            Dead oil viscosity in centipoise (cP).
        """
        density_standard = api_to_rho(api)
        y = np.power(10, 5.693 - 2.863 / density_standard)
        viscosity_standard = np.power(10, 0.505 * y / (17.8 + T)**1.163) - 1
        I = np.power(10, 18.6 * (0.1 * np.log10(viscosity_standard) + 1 / (np.log10(viscosity_standard) + 2)**0.1 - 0.985))
        return viscosity_standard + 0.145 * P * I

    @staticmethod
    def brine_properties(T, P, salinity):
        """
        Compute brine density and bulk modulus.

        Parameters
        ----------
        T : array_like
            Temperature in °C.
        P : array_like
            Pressure in MPa.
        salinity : array_like
            Salinity in ppm.

        Returns
        -------
        density : ndarray
            Brine density in g/cm³.
        bulk : ndarray
            Brine bulk modulus in GPa.
        """
        density = BatzleWang.brine_density(T, P, salinity)
        p_velocity = BatzleWang.brine_velocity(T, P, salinity)
        bulk = velocity_to_bulk(p_velocity, 0, density)
        return density, bulk

    @staticmethod
    def oil_properties(T, P, api, gor, gas_gravity):
        """
        Compute live oil density and bulk modulus.

        Parameters
        ----------
        T : array_like
            Temperature in °C.
        P : array_like
            Pressure in MPa.
        api : array_like
            API gravity.
        gor : array_like
            Gas-oil ratio in scf/STB.
        gas_gravity : array_like
            Gas specific gravity.

        Returns
        -------
        density : ndarray
            Oil density in g/cm³.
        bulk : ndarray
            Oil bulk modulus in GPa.
        """
        density = BatzleWang.oil_density(T, P, api, gor, gas_gravity)
        p_velocity = BatzleWang.oil_velocity(T, P, api, gor, gas_gravity)
        bulk = velocity_to_bulk(p_velocity, 0, density)
        return density, bulk

    @staticmethod
    def gas_properties(T, P, gas_gravity):
        """
        Compute natural gas density and bulk modulus.

        Parameters
        ----------
        T : array_like
            Temperature in °C.
        P : array_like
            Pressure in MPa.
        gas_gravity : array_like
            Gas specific gravity.

        Returns
        -------
        density : ndarray
            Gas density in g/cm³.
        bulk : ndarray
            Gas bulk modulus in GPa.
        """
        density = BatzleWang.gas_density(T, P, gas_gravity)
        p_velocity = BatzleWang.gas_velocity(T, P, gas_gravity)
        bulk = velocity_to_bulk(p_velocity, 0, density)
        return density, bulk

    @staticmethod
    def co2_properties(T, P, gas_gravity):
        """
        Compute CO₂ density and bulk modulus using Xu (2006) modification.

        Parameters
        ----------
        T : array_like
            Temperature in °C.
        P : array_like
            Pressure in MPa.
        gas_gravity : array_like
            Gas specific gravity.

        Returns
        -------
        density : ndarray
            CO₂ density in g/cm³.
        bulk : ndarray
            CO₂ bulk modulus in GPa.
        """
        density = BatzleWang.co2_density(T, P, gas_gravity)
        p_velocity = BatzleWang.co2_velocity(T, P, gas_gravity)
        bulk = velocity_to_bulk(p_velocity, 0, density)
        return density, bulk
