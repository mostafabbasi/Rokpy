"""
Empirical rock-physics relations.

This module provides a collection of commonly used empirical formulas and
convenience wrappers for estimating rock properties (velocities, density,
bulk/shear moduli, porosity and velocity-ratio transforms) from measurable
inputs such as porosity, clay content, pressure, and seismic velocities.
"""

import numpy as np
from rokpy import conversions
from rokpy.effective_medium import BoundMethods, FluidEffectMethods


class EmpiricalRelations:
    """Namespace of empirical relations to estimate rock properties.

    This container groups nested classes that implement commonly-used empirical
    formulas in rock physics for estimating P- and S-wave velocities, density,
    bulk/shear moduli, porosity and velocity-ratio transforms.

    Notes
    -----
    - Units: velocities in m/s, density in g/cc, moduli in GPa unless otherwise
      documented in the method docstrings.
    - Methods are implemented as @staticmethod to act as functional utilities and
      to be easily referenced without instantiation.
    """
    class PVelocity:
        """Functions to estimate P-wave velocity (Vp).

        Sub-namespaces:
        - FromSVelocity: invert or map Vs -> Vp using Castagna/Vernik relations.
        - FromDensity: invert density-velocity relations (Gardner, Brocher).
        - FromProperties: empirical relations using porosity, clay, pressure, etc.
        """
        class FromSVelocity:
            """Conversions from S-wave velocity (Vs) to P-wave velocity (Vp)."""

            @staticmethod
            def castagna_inverse(s_velocity, castagna_coefs):
                """Invert Castagna parabolic relation to estimate Vp from Vs.

                Relation: Vs = a*Vp^2 + b*Vp + c. This function returns the positive
                real root for Vp. If the quadratic coefficient is zero the linear
                inverse is used.

                Parameters
                ----------
                s_velocity : array_like
                    S-wave velocity (m/s).
                castagna_coefs : sequence
                    Coefficients [a, b, c] of Castagna polynomial.

                Returns
                -------
                ndarray or float
                    Estimated P-wave velocity (m/s).
                """
                if castagna_coefs[0] == 0:
                    p_velocity = 1/castagna_coefs[1] * s_velocity -castagna_coefs[2]/castagna_coefs[1]
                    return p_velocity
                else:
                    p_velocity = -castagna_coefs[1]/2/castagna_coefs[0] + np.sqrt(4*castagna_coefs[0]*(s_velocity- (castagna_coefs[2] - (castagna_coefs[1]**2)/4/castagna_coefs[0]) ))/2/castagna_coefs[0]
                    return p_velocity

            @staticmethod
            def vernik_inverse(s_velocity, vernik_coefs):
                """Invert Vernik squared-parabolic relation to estimate Vp from Vs.

                Vernik uses Vs^2 = a*Vp^4 + b*Vp^2 + c. This returns the positive
                Vp root. If a==0 the formula reduces to a quadratic in Vp^2.

                Parameters
                ----------
                s_velocity : array_like
                    S-wave velocity (m/s).
                vernik_coefs : sequence
                    Coefficients [a, b, c] for the Vernik squared-parabolic form.

                Returns
                -------
                ndarray or float
                    Estimated P-wave velocity (m/s).
                """
                if vernik_coefs[0] == 0:
                    p_velocity = np.sqrt(1/vernik_coefs[1] * s_velocity**2 -vernik_coefs[2]/vernik_coefs[1])
                    return p_velocity
                else:
                    p_velocity = np.sqrt(-vernik_coefs[1]/2/vernik_coefs[0] + np.sqrt(4*vernik_coefs[0]*(s_velocity**2 - (vernik_coefs[2] - (vernik_coefs[1]**2)/4/vernik_coefs[0]) ))/2/vernik_coefs[0] )
                    return p_velocity

        class FromDensity:
            """Relations to compute Vp from bulk density."""

            @staticmethod
            def gardner_inverse(density, gardner_coefs):
                """Invert Gardner power-law to estimate Vp from density.

                Gardner forward: rho = a * Vp^b.
                Inverse: Vp = (rho / a)^(1/b).

                Parameters
                ----------
                density : array_like
                    Bulk density (g/cc).
                gardner_coefs : sequence
                    [a, b] Gardner coefficients.

                Returns
                -------
                ndarray or float
                    Estimated P-wave velocity (m/s).
                """
                pvelocity = gardner_coefs[0]**(-1/gardner_coefs[1]) * density**(1/gardner_coefs[1])
                return pvelocity

            @staticmethod
            def brocher (density):
                """Brocher (1996) empirical polynomial giving Vp as function of density.

                Parameters
                ----------
                density : array_like
                    Bulk density (g/cc).

                Returns
                -------
                ndarray or float
                    P-wave velocity in m/s (Brocher polynomial is scaled to m/s).
                """
                p_velocity = (39.128*density - 63.064*density**2 + 37.083*density**3 - 9.1819*density**4 + 0.8228*density**5)*1000
                return p_velocity

        class FromProperties:
            """Empirical relations estimating Vp from rock/fluid/mineral properties."""

            @staticmethod
            def han (porosity, clay_content, han_coefs):
                """Han et al. multilinear relation for velocity.

                Vp = a + b*porosity + c*clay

                Parameters
                ----------
                porosity : array_like
                clay_content : array_like
                han_coefs : sequence
                    [a, b, c] coefficients.

                Returns
                -------
                ndarray or float
                    Estimated P-wave velocity (m/s).
                """
                velocity = han_coefs[0] + han_coefs[1]*porosity + han_coefs[2]*clay_content
                return velocity

            @staticmethod
            def eberhart_phillips (porosity, clay_content, effective_pressure):
                """Eberhart-Phillips empirical Vp relation including stress effect.

                Implementation follows common form scaled to m/s. effective_pressure
                expected in MPa (function converts to kbar internally).

                Parameters
                ----------
                porosity : array_like
                clay_content : array_like
                effective_pressure : array_like
                    Effective pressure in MPa.

                Returns
                -------
                ndarray or float
                    Estimated P-wave velocity (m/s).
                """
                effective_pressure = effective_pressure * 0.01 # MPa to Kbar conversion
                p_velocity = (5.77 - 6.94*porosity - 1.73*np.sqrt(clay_content) + 0.446*(effective_pressure-np.exp(-16.7*effective_pressure)))*1000
                return p_velocity

            @staticmethod
            def raymer(porosity, mineralset_pvelocity, fluidset_pvelocity):
                """Raymer empirical mixing estimate (approximate Voigt-type).

                Vp = (1-phi)^2 * Vp_mineral + phi * Vp_fluid

                Parameters
                ----------
                porosity : array_like
                mineralset_pvelocity : array_like
                fluidset_pvelocity : array_like

                Returns
                -------
                ndarray or float
                    Estimated P-wave velocity (m/s).
                """
                p_velocity = (1-porosity)**2*mineralset_pvelocity + porosity*fluidset_pvelocity
                return p_velocity

            @staticmethod
            def wyllie(porosity, mineralset_pvelocity, fluidset_pvelocity):
                """Wyllie time-average equation for P-wave velocity.

                1/Vp = phi/Vf + (1-phi)/Vm

                Parameters
                ----------
                porosity : array_like
                mineralset_pvelocity : array_like
                fluidset_pvelocity : array_like

                Returns
                -------
                ndarray or float
                    Estimated P-wave velocity (m/s).
                """
                p_velocity = 1/( porosity/fluidset_pvelocity + (1-porosity)/mineralset_pvelocity)
                return p_velocity

            @staticmethod
            def raymerdvorkin (porosity, mineralset_pvelocity, fluidset_pvelocity):
                """Raymer-Dvorkin style velocity estimator (placeholder variant).

                Currently implemented as the same expression as Raymer above.

                Parameters
                ----------
                porosity, mineralset_pvelocity, fluidset_pvelocity : array_like

                Returns
                -------
                ndarray or float
                    Estimated P-wave velocity (m/s).
                """
                p_velocity = (1-porosity)**2*mineralset_pvelocity + porosity*fluidset_pvelocity
                return p_velocity  
        
    class SVelocity:
        """Functions to estimate S-wave velocity (Vs)."""

        class FromPVelocity:
            """Methods computing Vs from Vp or mixed-mineral Castagna lines."""

            @staticmethod
            def castagna(p_velocity: np.ndarray, castagna_coefs: np.ndarray):
                """Castagna empirical polynomial: Vs = a*Vp^2 + b*Vp + c.

                Parameters
                ----------
                p_velocity : array_like
                    P-wave velocity (m/s).
                castagna_coefs : sequence
                    [a, b, c] coefficients.

                Returns
                -------
                ndarray or float
                    S-wave velocity (m/s).
                """
                s_velocity =  castagna_coefs[0]*p_velocity**2 + castagna_coefs[1]*p_velocity + castagna_coefs[2]*p_velocity**0
                return s_velocity

            @staticmethod
            def greenberg_castagna_100(p_velocity: np.ndarray, castagna_coef_list: list, fraction_list: list):
                """Greenberg & Castagna mixed-mineral Vs averaging.

                Evaluate Castagna Vs for each mineral line then mix with VRH/HRH
                averaging (Voigt-Reuss-Hill) using the provided fractions.

                Parameters
                ----------
                p_velocity : array_like
                castagna_coef_list : list of coefficient sequences
                fraction_list : list of fractions for each mineral

                Returns
                -------
                ndarray or float
                    Mixed S-wave velocity (m/s).
                """
                s_velocity_list = []
                for coefs in castagna_coef_list:
                    s_velocity_list.append(EmpiricalRelations.SVelocity.FromPVelocity.castagna(p_velocity, coefs))
                s_velocity = BoundMethods.voigt_reuss_hill_average(s_velocity_list, fraction_list, upper_weight=0.5)
                return  s_velocity

            @staticmethod
            def greenberg_castagna(p_velocity: np.ndarray, 
                                bulk_density: np.ndarray, 
                                fraction_list: list[np.ndarray], 
                                castagna_coef_list: list[np.ndarray], 
                                porosity: np.ndarray, 
                                mineralset_bulk: np.ndarray, 
                                fluidset_bulk: np.ndarray, 
                                brine_bulk: np.ndarray, 
                                brine_density: np.ndarray, 
                                fluidset_density: np.ndarray, 
                                No_of_iterations: int):
                """Iterative estimator combining Castagna Vs lines with Gassmann consistency.

                The method perturbs Vp and searches for a combination of Vp/Vs at 100%
                saturation that is consistent with Gassmann saturation substitutions.
                Returns Vs estimated from a best-fit saturated dry-rock pair.

                Parameters
                ----------
                p_velocity : array_like
                bulk_density : array_like
                fraction_list, castagna_coef_list : lists describing mineral mix
                porosity : array_like
                mineralset_bulk, fluidset_bulk, brine_bulk : modulus arrays
                brine_density, fluidset_density : density arrays
                No_of_iterations : int
                    Number of random perturbation iterations for the search.

                Returns
                -------
                ndarray or float
                    Estimated S-wave velocity (m/s) at 100% brine saturation.
                """
                Nsample = np.size(p_velocity)
                fraction_list = np.array(fraction_list)

                bulk_density_100 = bulk_density + (brine_density-fluidset_density)*porosity

                bestdiff = np.inf
                pvelocity_100 = p_velocity
                svelocity_100 = p_velocity/2
                for iter in range(No_of_iterations):
                    delta = 0.1*np.random.uniform()
                    pvelocity100_rnd = (1+delta)*p_velocity
                    svelocity100_rnd = EmpiricalRelations.SVelocity.FromPVelocity.greenberg_castagna_100(pvelocity100_rnd, castagna_coef_list, fraction_list)
                    shear100_rnd = svelocity100_rnd**2 * bulk_density_100
                    
                    Gsat_rnd = shear100_rnd
                    Ksat_rnd = p_velocity**2 * bulk_density - 4/3*Gsat_rnd
                    
                    shear_100_rndnew = Gsat_rnd
                    bulk_100_rndnew = FluidEffectMethods.gassmann_substitution(Ksat_rnd, Gsat_rnd, mineralset_bulk, fluidset_bulk, brine_bulk, porosity)
                    
                    pvelocity_100_rndnew, _ = conversions.modulus_to_pvelocity(bulk_100_rndnew, shear_100_rndnew, bulk_density_100)
                    currdiff = np.linalg.norm(pvelocity_100_rndnew - pvelocity100_rnd, ord=2)
                    
                    if currdiff< bestdiff:
                        bestdiff = currdiff
                        pvelocity_100 = pvelocity100_rnd
                        svelocity_100 = svelocity100_rnd

                bulk_100, shear_100 = conversions.velocity_to_modulus(pvelocity_100, svelocity_100, bulk_density_100)
                # Ksat, Gsat = FluidEffectMethods.gassmann_substitution( bulk_100, shear_100, brine_bulk, fluidset_bulk, porosity)
                return conversions.modulus_to_svelocity(shear_100, bulk_density)

            @staticmethod
            def vernik(p_velocity: np.ndarray, vernik_coefs):
                """Vernik squared-parabolic Vs(Vp): Vs = sqrt(a*Vp^4 + b*Vp^2 + c).

                Parameters
                ----------
                p_velocity : array_like
                coefficients : sequence
                    [a, b, c] Vernik coefficients.

                Returns
                -------
                ndarray or float
                    S-wave velocity (m/s).
                """
                s_velocity = np.sqrt( vernik_coefs[0]*p_velocity**4 + vernik_coefs[1]*p_velocity**2 + vernik_coefs[2]*p_velocity**0 )
                return s_velocity

            @staticmethod
            def krief(p_velocity, mineralset_pvelocity, mineralset_svelocity, fluidset_pvelocity, critical_porosity = 1):
                """Krief-style Vs estimator derived from Reuss lower bound and scaling.

                Uses a Reuss average for Vp to constrain Vs by scaling the mineral Vs.

                Parameters
                ----------
                p_velocity : array_like
                mineralset_pvelocity, mineralset_svelocity, fluidset_pvelocity : array_like
                critical_porosity : float

                Returns
                -------
                ndarray or float
                    Estimated S-wave velocity (m/s).
                """
                VpR = BoundMethods.reuss([mineralset_pvelocity, fluidset_pvelocity], [1-critical_porosity, critical_porosity])
                s_velocity = np.sqrt( (p_velocity**2 - VpR**2) / (mineralset_pvelocity**2 - VpR**2) * mineralset_svelocity**2 )
                return s_velocity

            @staticmethod
            def vernik_sourcerock(p_velocity, TOC, TOCref, aref, a0=-0.22, b=0.58):
                """Vernik source-rock Vs estimator that includes TOC scaling.

                Empirical linearization mapping Vp and total organic carbon (TOC) to Vs.

                Parameters
                ----------
                p_velocity : array_like
                TOC : array_like
                    Total organic carbon (weight fraction).
                TOCref : float
                    Reference TOC used for normalization.
                aref, a0, b : floats
                    Empirical constants.

                Returns
                -------
                ndarray or float
                    S-wave velocity (m/s).
                """
                svelocity = 1000*( b*p_velocity/1000 + a0 + (aref-a0)*TOC/TOCref )
                return svelocity

        class FromProperties:
            """Methods to compute Vs from porosity, clay content and effective stress."""

            @staticmethod
            def han (porosity, clay_content, han_coefs):
                """Han multilinear form for Vs: Vs = a + b*phi + c*clay."""
                velocity = han_coefs[0] + han_coefs[1]*porosity + han_coefs[2]*clay_content
                return velocity

            @staticmethod
            def eberhart_phillips (porosity, clay_content, effective_pressure):
                """Eberhart-Phillips empirical Vs relation including stress dependence.

                effective_pressure expected in MPa (function converts to kbar internally).

                Returns Vs in m/s.
                """
                effective_pressure = effective_pressure * 0.01 # MPa to Kbar conversion
                s_velocity = (3.70 - 4.94*porosity - 1.57*np.sqrt(clay_content) + 0.361*(effective_pressure-np.exp(-16.7*effective_pressure)))*1000
                return s_velocity

            @staticmethod
            def raymerdvorkin (porosity, mineralset_svelocity, mineralset_density, fluidset_density):
                """Raymer-Dvorkin style Vs estimator using mass-weighted scaling.

                Note: 'fluidset_sensity' appears to be a typo in original code and
                represents fluid density used in the estimator.

                Parameters
                ----------
                porosity, mineralset_svelocity, mineralset_density, fluidset_sensity : array_like

                Returns
                -------
                ndarray or float
                    Estimated S-wave velocity (m/s).
                """
                rho = (1-porosity)*mineralset_density + porosity*fluidset_density
                Vs = (1-porosity)**2*mineralset_svelocity * np.sqrt((1-porosity)*mineralset_density/rho)
                return Vs
            
    class Density:
        """Functions to compute or convert bulk density from other properties."""

        class FromPVelocity:
            """Density estimators driven by seismic velocity."""

            @staticmethod
            def gardner (p_velocity, gardner_coefs):
                """Gardner forward relation: rho = a * Vp^b.

                Parameters
                ----------
                p_velocity : array_like
                    P-wave velocity (m/s).
                gardner_coefs : sequence
                    [a, b] coefficients.

                Returns
                -------
                ndarray or float
                    Bulk density in g/cc.
                """
                density = gardner_coefs[0] * p_velocity**gardner_coefs[1]
                return density

    class Bulk:
        """Helpers to estimate bulk modulus from porosity and constituent moduli."""

        class FromProperties:
            @staticmethod
            def vernik_drysoftsand(porosity, mineralset_bulk, mineralset_shear, poreshape_factor):
                """Vernik dry soft-sand bulk modulus estimator.

                Uses an empirical pore-shape factor to compute M_dry then subtracts
                the shear contribution to retrieve bulk modulus.

                Parameters
                ----------
                porosity : array_like
                mineralset_bulk : array_like
                    Mineral frame bulk modulus (GPa).
                mineralset_shear : array_like
                    Mineral frame shear modulus (GPa).
                poreshape_factor : float

                Returns
                -------
                ndarray or float
                    Dry-rock bulk modulus (GPa).
                """
                p = 3.6 + poreshape_factor*porosity
                q = 3.6 + poreshape_factor*porosity
                Mm = mineralset_bulk + 4/3*mineralset_shear
                Mdry = Mm/ (1 + p*porosity/(1-porosity))
                dry_shear = mineralset_shear/ (1 + q*porosity/(1-porosity))
                dry_bulk = Mdry - 4/3*dry_shear
                return dry_bulk

            @staticmethod
            def vernik_drysandstone(porosity, mineralset_bulk, mineralset_shear, poreshape_factor, effective_stress):
                """Vernik dry sandstone bulk modulus including crack density and stress.

                Parameters
                ----------
                porosity : array_like
                mineralset_bulk, mineralset_shear : array_like (GPa)
                poreshape_factor : float
                effective_stress : array_like (MPa)

                Returns
                -------
                ndarray or float
                    Dry-rock bulk modulus (GPa).
                """
                num = conversions.modulus_to_poisson(mineralset_bulk, mineralset_shear)
                ep0 = 0.3 + 1.6*porosity #Crack density at zero effective stress
                d = 0.07
                A = 16/9 * (1-num**2)/(1-2*num)
                p = 3.6 + poreshape_factor*porosity
                dry_bulk = mineralset_bulk/ (1 + p*porosity/(1-porosity) + A*ep0*np.exp(-d*effective_stress)/(1-porosity) )
                return dry_bulk

            @staticmethod
            def geertsma_dryrock(porosity, mineralset_bulk):
                """Simple Geertsma dry-rock bulk modulus approximation.

                M_dry = M_mineral / (1 + 50*phi)

                Parameters
                ----------
                porosity : array_like
                mineralset_bulk : array_like (GPa)

                Returns
                -------
                ndarray or float
                    Dry bulk modulus (GPa).
                """
                dry_bulk = mineralset_bulk/(1+50*porosity)
                return dry_bulk

    class Shear:
        """Functions to estimate dry-rock shear modulus from porosity and stress."""

        class FromProperties:
            @staticmethod
            def vernik_drysandstone(porosity, mineralset_bulk, mineralset_shear, poreshape_factor, effective_stress):
                """Vernik dry sandstone shear modulus with stress-dependent crack closure.

                Parameters
                ----------
                porosity, mineralset_bulk, mineralset_shear, poreshape_factor, effective_stress : array_like

                Returns
                -------
                ndarray or float
                    Dry-rock shear modulus (GPa).
                """
                num = conversions.modulus_to_poisson(mineralset_bulk, mineralset_shear)
                ep0 = 0.3 + 1.6*porosity #Crack density at zero effective stress
                d = 0.07
                B = 32/45 * (1-num)*(5-num)*(2-num)
                q = 3.6 + poreshape_factor*porosity
                dry_shear = mineralset_shear/ (1 + q*porosity/(1-porosity) + B*ep0*np.exp(-d*effective_stress)/(1-porosity) )
                return dry_shear

            @staticmethod
            def vernik_drysoftsand(porosity, mineralset_shear, poreshape_factor):
                """Vernik dry soft-sand shear modulus estimator.

                Parameters
                ----------
                porosity : array_like
                mineralset_shear : array_like (GPa)
                poreshape_factor : float

                Returns
                -------
                ndarray or float
                    Dry shear modulus (GPa).
                """
                q = 3.6 + poreshape_factor*porosity
                dry_shear = mineralset_shear/ (1 + q*porosity/(1-porosity))
                return dry_shear

    class Porosity:
        """Porosity estimators derived from seismic velocities and mixing laws."""

        class FromPVelocity:
            @staticmethod
            def raymer_inverse(p_velocity, mineralset_pvelocity, fluidset_pvelocity):
                """Raymer porosity inversion from velocity (placeholder).

                Note: original implementation references 'porosity' in the right-hand
                side before it is defined — this function likely needs correction
                if used. Kept docstring to indicate intended behavior:

                Estimates porosity from Vp using a rearranged Wyllie/Raymer relation.

                Parameters
                ----------
                p_velocity : array_like
                mineralset_pvelocity : array_like
                fluidset_pvelocity : array_like

                Returns
                -------
                ndarray or float
                    Estimated porosity (unitless fraction).
                """
                porosity  = ((2*mineralset_pvelocity - fluidset_pvelocity) - np.sqrt( (2*mineralset_pvelocity - fluidset_pvelocity)**2 - 4*mineralset_pvelocity*(mineralset_pvelocity - p_velocity) )) / (2*mineralset_pvelocity)
                return porosity

    class Modulus:
        """Miscellaneous modulus estimators and transforms."""

        class FromProperties:
            @staticmethod
            def nur_dryclastic(porosity, mineralset_modulus, phic): #Inputs must be ndarray
                """NUR dry-clastic modulus: linear decrease with porosity until critical phi.

                M_dry = M_mineral * (1 - phi / phic); values set to 0 for phi>phic.

                Parameters
                ----------
                porosity : ndarray
                mineralset_modulus : ndarray
                phic : float

                Returns
                -------
                ndarray
                    Dry-rock compressibility modulus (same units as mineralset_modulus).
                """
                Mdry = mineralset_modulus*(1-porosity/phic)
                Mdry[porosity>phic]=0
                return Mdry

            @staticmethod
            def krief_dryrock(mineralset_modulus, biot_coef):
                """Krief empirical dry-rock modulus scaling: M_dry = M_mineral * (1 - biot).

                Parameters
                ----------
                mineralset_modulus : array_like
                biot_coef : array_like or float

                Returns
                -------
                ndarray or float
                    Dry-rock modulus (same units as mineralset_modulus).
                """
                dry_modulus = mineralset_modulus*(1-biot_coef)
                return dry_modulus

    class VelocityRatio:
        """Empirical relations and transforms for Vp/Vs (velocity ratio)."""

        class FromPVelocity:
            @staticmethod
            def ludwig (p_velocity):
                """Ludwig empirical Poisson's ratio polynomial mapped to Vp/Vs.

                Uses an approximate polynomial for Poisson's ratio (nu) from Vp
                (Brocher/Ludwig style) then converts to Vp/Vs ratio.

                Parameters
                ----------
                p_velocity : array_like
                    P-wave velocity in m/s.

                Returns
                -------
                ndarray or float
                    Vp/Vs ratio (unitless).
                """
                p_velocity = p_velocity/1000
                nu =  0.769 - 0.226*p_velocity + 0.0316*p_velocity**2 - 0.0014*p_velocity**3
                Vp2Vs = conversions.poisson_to_velratio(nu)
                return Vp2Vs

            @staticmethod
            def brocher(p_velocity):
                """Brocher empirical Poisson's ratio polynomial mapped to Vp/Vs.

                Parameters
                ----------
                p_velocity : array_like
                    P-wave velocity in m/s.

                Returns
                -------
                ndarray or float
                    Vp/Vs ratio (unitless).
                """
                p_velocity = p_velocity/1000
                nu =  0.8835 - 0.315*p_velocity + 0.0491*p_velocity**2 - 0.0024*p_velocity**3
                Vp2Vs = conversions.poisson_to_velratio(nu)
                return Vp2Vs
