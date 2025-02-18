#!/usr/bin/env python2.7
# -*- coding: iso-8859-1 -*-

'''
Calculates soil parameters from the soil textural class or from the fraction
of sand, silt and clay in the mineral soil component.
The parameters can also be adjusted for a organic soil component.

 - Karina Williams and Heather Ashton

'''
from __future__ import print_function

import math
import argparse
import numpy as np
import scipy.special

'''
References
  * [Cosby 1984]
    Cosby, B.J., Hornberger, G.M., Clapp, R.B. and Ginn, T.T., 1984. A
    statistical exploration of the relationships of soil moisture
    characteristics to the physical properties of soils. Water Resources
    Research. 20. 682-690.
    http://onlinelibrary.wiley.com/doi/10.1029/WR020i006p00682/pdf
    Note that logs in this paper are log to the base 10.
  * [AncilDoc_Cosby]
    AncilDoc_Cosby.html in the ancil documentation (see attachment to
    https://code.metoffice.gov.uk/trac/ancil/ticket/250)
  * [AncilDoc_SoilThermal]
    AncilDoc_SoilThermal.html in the ancil documentation (see attachment to
    https://code.metoffice.gov.uk/trac/ancil/ticket/250)
  * [Dankers 2011]
    R. Dankers, E. J. Burke, J. Price, Simulation of permafrost and seasonal
    thaw depth in the JULES land surface scheme
    www.the-cryosphere.net/5/773/2011/
  * [Chadburn 2015]
    S. Chadburn, E. Burke, R. Essery, J. Boike, M. Langer, M. Heikenfeld,
    P. Cox, and P. Friedlingstein, An improved representation of physical
    permafrost dynamics in the JULES land surface model
    www.geosci-model-dev-discuss.net/8/715/2015/
  * [Best 2011]
    M. J. Best, M. Pryor, D. B. Clark, G. G. Rooney, Essery, C. B. Ménard, 
    J. M. Edwards, M. A. Hendry, A. Porson, N. Gedney, L. M. Mercado, S. Sitch, 
    E. Blyth, O. Boucher, P. M. Cox, C. S. B. Grimmond, R. J. Harding
    The Joint UK Land Environment Simulator (JULES), model description - Part 1: Energy and water fluxes
    http://www.geosci-model-dev.net/4/677/2011/
  * [Farouki 1981]
    Omar T. Farouki, The thermal properties of soils in cold regions
    Cold Regions Science and Technology, Vol. 5, No. 1. (September 1981), pp. 67-75,
    doi:10.1016/0165-232x(81)90041-0
'''

__version__ = "last known fcm revision: " + ''.join(filter(str.isdigit,
                                                   "$Rev: 8555 $"))


# array contains percentage silt, percentage sand, percentage clay from
# table 2 of [Cosby 1984]
# plus silt (see email from Heather Ashton 6.7.15)
SOIL_TEXTURAL_CLASS = {
    'Sand':            [ 5, 92,  3],
    'Loamy sand':      [12, 82,  6],
    'Sandy loam':      [32, 58, 10],
    'Loam':            [39, 43, 18],
    'Silty loam':      [70, 17, 13],
    'Sandy clay loam': [15, 58, 27],
    'Clay loam':       [34, 32, 34],
    'Silty clay loam': [56, 10, 34],
    'Sandy clay':      [ 6, 52, 42],
    'Silty clay':      [47,  6, 47],
    'Clay':            [20, 22, 58],
    'Silt':            [90,  7,  3]
}


# Parameters for organic soil used in the SOC experiment
# from [Dankers 2011] table 2. Layers are 0-10cm, 10-35cm, 35-100cm
ORGANIC_SOIL = {'b': [2.7, 6.1, 12.0],
                # b, Brooks-Corey exponent in the soil hydraulic
                # characteristics
                'sathh': [0.0103, 0.0102, 0.0101],
                # Psi_s, absolute value of the soil matric suction at
                # saturation, m
                'satcon': [0.28, 0.002, 0.0001],
                # K_s, hydraulic conductivity at saturation, kg m-2 s-1
                'sm_sat': [0.93, 0.88, 0.83],
                # theta_s, volumetric soil moisture content at saturation,
                # m3 water per m3 soil
                'sm_crit': [0.11, 0.34, 0.51],
                # theta_c, volumetric soil moisture content at the critical
                # point at which soil moisture stress starts to reduce
                # transpiration, m3 water per m3 soil
                'sm_wilt': [0.03, 0.18, 0.37],
                # theta_w, volumetric soil moisture content at the wilting
                # point, m3 water per m3 soil
                'hcap': [0.58E+6, 0.58E+6, 0.58E+6],
                # c, dry heat capacity, J m-3 K-1
                'hcon': [0.06, 0.06, 0.06]
                # lambda, dry thermal conductivity, W m-1 K-1
                }


COSBY_PARAMETER_DESCRIPTION = {
                'b': 'Brooks-Corey exponent in the soil hydraulic characteristics, dimensionless',
                'sathh': 'absolute value of the soil matric suction at saturation (Psi_s), in m',
                'satcon': 'hydraulic conductivity at saturation (K_s), in kg m-2 s-1',
                'sm_sat': 'volumetric soil moisture content at saturation (theta_s), in m3 water per m3 soil',
                'sm_crit': 'volumetric soil moisture content at the critical point at '
                           'which soil moisture stress starts to reduce transpiration (sm_crit), in m3 water per m3 soil',
                'sm_wilt': 'volumetric soil moisture content at the wilting point (sm_wilt), in m3 water per m3 soil',
                'hcap': 'dry heat capacity (c), in J m-3 K-1',
                'hcon': 'dry thermal conductivity (lambda), in W m-1 K-1',
                'albsoil': 'soil albedo (not calculated by this script)',
                'clay': 'soil clay fraction (g clay per g soil)',
                }


RHO_W = 999.97  # density of water kg m-3 (at 4 degC)
G_GRAVITY = 9.81  # acceleration due to gravity in m s-2


# calculated in soil_van_genuchten_parameters.py
PARAMETERS_HWSD_VG_VERY_FINE = {
                'b':11.1982082867,
                'sathh':0.324044069994,
                'satcon':0.00152,
                'sm_sat':0.4559,
                'sm_crit':0.367653508572,
                'sm_wilt':0.263058871608,
                'clay':0.52,
                'silt':0.21,
                'sand':0.27,
                }


class SoilType(object):
    ''' defines the properties of the soil '''
    def __init__(self, soil_textural_class=None, f_clay=None, f_sand=None,
                 f_silt=None, f_organic=None, soc_layer=None,
                 l_cap_cutoffs=False, empty=False):
        '''
        Kwargs:

         - f_clay : fraction of clay in mineral soil
         - f_sand : fraction of sand in mineral soil
         - f_silt : fraction of silt in mineral soil
         - soil_textural_class : soil textural class as in table 2 of
           [Cosby 1984]
         - f_organic : fraction of organic matter in soil
         - soc_layer : layer to take organic properties from (described in
           [Dankers 2011] table 2)
         - l_cap_cutoffs: whether to apply the CAP cutoffs, which set 
           very fine (f_clay >= 0.6 ) or organic soils (f_organic >= 0.1 )
           to values from (or calculated from) the HWSD look up table used by CAP.
         - empty: if False, requires f_clay, f_sand, f_silt or
           soil_textural_class and then will go ahead and fill the other fields.
           If True, creates an object that just contains an empty 
           soil_textural_class dictionary.
        '''

        if empty:
            self.jules_soil_parameters = {}
            return 

        fracs_are_missing = [f is None for f in [f_clay, f_sand, f_silt]]

        if soil_textural_class is None:
            if any(fracs_are_missing):
                raise ValueError('need to give either soil texture fractions '
                                 '(f_clay, f_sand, f_silt) or '
                                 'soil_textural_class')
        else:
            if not all(fracs_are_missing):
                raise ValueError('should not give soil texture fractions '
                                 '(f_clay, f_sand, f_silt) if '
                                 'soil_textural_class is given')

            if soil_textural_class not in SOIL_TEXTURAL_CLASS:
                raise ValueError('soil_textural_class sould be one of'
                                 + str(list(SOIL_TEXTURAL_CLASS.keys())))

            [f_silt_percent, f_sand_percent, f_clay_percent] = \
                SOIL_TEXTURAL_CLASS[soil_textural_class]

            f_silt = float(f_silt_percent)/100.0
            f_sand = float(f_sand_percent)/100.0
            f_clay = float(f_clay_percent)/100.0

        self.f_clay = f_clay
        self.f_sand = f_sand
        self.f_silt = f_silt

        if np.max(abs(1.0 - (self.f_clay + self.f_sand + self.f_silt))) > 1.0E-3:
            raise ValueError('soil texture fractions should add up to '
                             'approx 1')

        self.jules_soil_parameters = {}
        self.fill_cosby_soil_parameters()
        self.fill_heat_capacity_of_soil()
        self.fill_thermal_conductivity_of_soil()

        if f_organic is None:
            self.f_organic = 0.0 
        else:    
            if not (0.0 <= f_organic <= 1.0):
                raise ValueError('if given, f_organic should be 0.0 to 1.0')
            self.f_organic = f_organic
            
        if l_cap_cutoffs:
            if soc_layer is not None:
                raise ValueError('if apply_cap_cutoffs is True, soc_layer should not be given')     
            
            self.apply_cap_cutoffs()
        else:                
            if self.f_organic > 0.0:
                
                if soc_layer not in [1, 2, 3]:
                    raise ValueError('soc_layer not recognised. '
                                    'Should be 1, 2 or 3.')

                self.soc_layer = soc_layer

                self.adapt_soil_parameters_for_organic_content()
                
    def apply_cap_cutoffs(self):
        '''
        Apply the CAP cutoffs, which set very fine (f_clay >= 0.6 ) 
        or organic soils (f_organic >= 0.1 ) to values from (or calculated from) 
        the HWSD look up table used by CAP.
        
        This function has not been generalised for arrays.
        '''
        
        var_list = ['b', 'sathh', 'satcon', 'sm_sat',
                    'sm_crit', 'sm_wilt', 'clay']
                    
        if self.f_clay >= 0.6 or self.f_organic >= 0.1:
            for key in var_list:
                self.jules_soil_parameters[key] = PARAMETERS_HWSD_VG_VERY_FINE[key]
            
            # fracs are reset to class averages in CAP file hwsd_soil.F
            self.f_clay = PARAMETERS_HWSD_VG_VERY_FINE['clay']
            self.f_sand = PARAMETERS_HWSD_VG_VERY_FINE['sand']
            self.f_silt = PARAMETERS_HWSD_VG_VERY_FINE['silt']
            self.f_organic = np.nan # not sure what this should be so setiing to NaN for the moment
                
            self.fill_heat_capacity_of_soil()
            self.fill_thermal_conductivity_of_soil()
        
        return        

    def fill_cosby_soil_parameters(self):
        '''
        Calculate soil parameters for non-organic soil
        using [AncilDoc_Cosby], which references [Cosby 1984]
        '''

        # Clapp Hornberger parameter b
        # JULES_SOIL_PROPS units: dimensionless.
        self.jules_soil_parameters['b'] = \
            3.10 + 15.70 * self.f_clay - 0.3 * self.f_sand

        # Saturated soil water suction SATHH in terms of log to the base 10
        # (n.b. used to use nat log)
        # JULES_SOIL_PROPS units: m. (says 'the *absolute* value of the soil
        # matric suction at saturation')
        self.jules_soil_parameters['sathh'] = \
            0.01 * (10.0 ** (2.17 - 0.63 * self.f_clay - 1.58 * self.f_sand))

        # Saturated hydrological soil conductivity Ks in terms of log to the
        # base 10 (n.b. used to use nat log)
        # JULES_SOIL_PROPS units: kg m^-2 s^-1
        self.jules_soil_parameters['satcon'] = \
            10.0 ** (-2.75 - 0.64 * self.f_clay + 1.26 * self.f_sand)

        # Volumetric soil water concentration at saturation point theta_s
        # JULES_SOIL_PROPS units: m^3 water per m^3 soil
        self.jules_soil_parameters['sm_sat'] =  \
            0.505 - 0.037 * self.f_clay - 0.142 * self.f_sand

        # Volumetric soil moisture concentration at wilting point theta_w
        # This is calculated assuming a matric water potential (psi) of 1.5MPa
        # 1.0 MPa = 1.0E6 kg m^-1 s^-2
        # JULES_SOIL_PROPS units: m^3 water per m^3 soil
        self.jules_soil_parameters['sm_wilt'] = self._theta(psi=1.5E6)

        # Volumetric soil moisture concentration at critical point theta_c
        # This is calculated assuming a matric water potential (psi)
        # of 0.033MPa.
        # JULES_SOIL_PROPS units: m^3 water per m^3 soil
        self.jules_soil_parameters['sm_crit'] = self._theta(psi=0.033E6)
        
        # needed for l_triffid=T
        self.jules_soil_parameters['clay'] = self.f_clay
        
        # albsoil is not generated by this file - set to NaN
        self.jules_soil_parameters['albsoil'] = np.nan

    def _theta(self, psi=None):
        '''
        Wrapper to the Brooks and Corey equation for calculating vol.
        soil moisture.
        '''
        theta = brooks_and_corey_equation(
            abs_psi=psi, 
            theta_sat=self.jules_soil_parameters['sm_sat'], 
            theta_res=0.0, 
            b=self.jules_soil_parameters['b'], 
            sathh=self.jules_soil_parameters['sathh'])

        return theta

    def fill_thermal_conductivity_of_soil(self):
        '''
        Calculate thermal conductivity of non-organic soil
        using [AncilDoc_SoilThermal] Method 1 [Farouki 1981].
        This method is the default option in CAP.
        Kier Bovis, (email, Jan 2018), says that he always uses this
        option and that it "appear[s] to be always used" in CAP.
        '''

        lambda_air  = 0.025  # W m^-1 K^-1
        lambda_clay = 1.16   # W m^-1 K^-1
        lambda_sand = 1.57   # W m^-1 K^-1
        lambda_silt = 1.57   # W m^-1 K^-1

        a = (1.0 - self.jules_soil_parameters['sm_sat']) * self.f_clay
        b = (1.0 - self.jules_soil_parameters['sm_sat']) * self.f_sand
        c = (1.0 - self.jules_soil_parameters['sm_sat']) * self.f_silt

        self.jules_soil_parameters['hcon'] = (
            lambda_air ** self.jules_soil_parameters['sm_sat']
            * lambda_clay ** a * lambda_sand ** b * lambda_silt ** c
            )

        return

    def fill_heat_capacity_of_soil(self):
        '''
        Calculate heat capacity of non-organic soil
        using [AncilDoc_SoilThermal] Method 1 
        This method is the default option in CAP.
        Kier Bovis, (email, Jan 2018), says that he always uses this
        option and that it "appear[s] to be always used" in CAP.
        '''

        c_clay = 2.373E6  # J m^-3 K^-1
        c_sand = 2.133E6  # J m^-3 K^-1
        c_silt = 2.133E6  # J m^-3 K^-1

        self.jules_soil_parameters['hcap'] = (
            (1.0 - self.jules_soil_parameters['sm_sat'])
            * (self.f_clay * c_clay + self.f_sand * c_sand +
               self.f_silt * c_silt)
            )

        return

    def adapt_soil_parameters_for_organic_content(self):
        '''
        Adapts the soil parameters (calculated for non-organic)
        soil) to include organic content.
        '''

        i = self.soc_layer - 1

        # [Chadburn 2015] Equation A1
        self.jules_soil_parameters['b'] = (
            (1.0 - self.f_organic) * self.jules_soil_parameters['b'] +
            self.f_organic * ORGANIC_SOIL['b'][i]
            )
        # [Chadburn 2015] Equation A2
        self.jules_soil_parameters['sathh'] = (
            self.jules_soil_parameters['sathh'] ** (1.0 - self.f_organic) *
            ORGANIC_SOIL['sathh'][i] ** self.f_organic
            )
        # [Chadburn 2015] Equation A3
        self.jules_soil_parameters['satcon'] = (
            self.jules_soil_parameters['satcon'] ** (1.0 - self.f_organic) *
            ORGANIC_SOIL['satcon'][i] ** self.f_organic
            )
        # [Chadburn 2015] Equation A4
        self.jules_soil_parameters['sm_sat'] = (
            (1.0 - self.f_organic) * self.jules_soil_parameters['sm_sat'] +
            self.f_organic * ORGANIC_SOIL['sm_sat'][i]
            )
        # [Chadburn 2015] Equation A5
        # slight difference in psi to be consistent with the non-organic
        # calculation
        self.jules_soil_parameters['sm_crit'] = self._theta(psi=0.033E6)

        # [Chadburn 2015] Equation A6
        # slight difference in psi to be consistent with the non-organic
        # calculation
        self.jules_soil_parameters['sm_wilt'] = self._theta(psi=1.5E6)

        # [Chadburn 2015] Equation A7
        self.jules_soil_parameters['hcap'] = (
            (1.0 - self.f_organic) * self.jules_soil_parameters['hcap'] +
            self.f_organic * ORGANIC_SOIL['hcap'][i]
            )
        # [Chadburn 2015] Equation A8
        self.jules_soil_parameters['hcon'] = (
            self.jules_soil_parameters['hcon'] ** (1.0 - self.f_organic) *
            ORGANIC_SOIL['hcon'][i] ** self.f_organic
            )

    def print_jules_soil_parameters(self):
        '''
        prints soil parameters in a format that
        can be copy-and-pasted into the JULES_SOIL_PROPS
        namelist
        '''

        var = ''
        const_val = ''
        
        # Order matters because we want clay last - it's not needed in all runs
        parameter_list = ['b', 'sathh', 'satcon', 'sm_sat', 'sm_crit', 'sm_wilt', 'hcap', 'hcon', 'albsoil', 'clay']
        
        for key in parameter_list:
            var = var + ("'" + key + "'").rjust(14)
            const_val = const_val + '    ' + \
                ("%10.4G" % (self.jules_soil_parameters[key]))

        print(var)
        print(const_val)

        return


def brooks_and_corey_equation(abs_psi=None, theta_sat=None, theta_res=None, b=None, sathh=None):
    '''
    Calculates the volumetric soil moisture concentration (theta) at a
    matric water potential of psi. Works with numpy arrays if they are broadcastable.

    psi is in kg m^-1 s^-2 (i.e Pa). Positive.
    b is the Brooks Corey exponent
    theta_res is the residual vol. soil moisture in m^3 water per m^3 soil
    theta_sat is the saturated vol. soil moisture in m^3 water per m^3 soil
    sathh is the *absolute* value of the soil matric suction at saturation in m
    theta is vol. soil moisture in m^3 water per m^3 soil    

    note: this differs from [AncilDoc_Cosby]
          because has extra factor sm_sat
          (checked this with Heather Ashton 1.4.15)
          and also theta_res is not assumed zero.
          
    See also [Best 2011] equation 58 (where theta_res is implicitly subtracted from theta_sat and theta).      
    '''

    # h is the soil water pressure head
    h = abs_psi / (RHO_W * G_GRAVITY)

    theta = theta_res + \
        ( theta_sat - theta_res) * (
        (sathh / h)
        ** (1.0 / b)
        )

    return theta


def inverse_brooks_and_corey_equation(theta=None, theta_sat=None, theta_res=None, b=None, sathh=None):
    '''
    Calculates the matric water potential (psi) at a volumetric 
    soil moisture concentration (theta). 
    Works with numpy arrays if they are broadcastable.

    abs_psi is in kg m^-1 s^-2 (i.e Pa). Positive.
    b is the Brooks Corey exponent
    theta_res is the residual vol. soil moisture in m^3 water per m^3 soil
    theta_sat is the saturated vol. soil moisture in m^3 water per m^3 soil
    sathh is the *absolute* value of the soil matric suction at saturation in m
    theta is vol. soil moisture in m^3 water per m^3 soil    
    '''

    abs_psi = sathh \
        * ( (theta - theta_res) / (theta_sat - theta_res) ) \
        ** (- b) * (RHO_W * G_GRAVITY) 

    return abs_psi


def hydraulic_conductivity_brooks_corey(theta=None, theta_sat=None, theta_res=None, b=None, satcon=None):
    '''
    Calculates the hydraulic conductivity using the Brooks and Corey
    method. See [Best 2011] equation 59.

    theta is vol. soil moisture in m^3 water per m^3 soil    
    theta_res is the residual vol. soil moisture in m^3 water per m^3 soil
    satcon is saturated hydraulic conductivity in kg m-2 s-1
    b is the Brooks Corey exponent
    hyd_con is in kg m-2 s-1
    '''

    hyd_con = satcon\
        * ( (theta - theta_res) / (theta_sat - theta_res) ) \
        ** (2.0 * b + 3.0) 

    return hyd_con


def soil_texture_from_sm_wilt_sm_crit(sm_wilt=None, sm_crit=None, psi_wilt=1.5E6, psi_crit=0.033E6):

    b = calc_b_from_sm_wilt_sm_crit(
        sm_wilt=sm_wilt, sm_crit=sm_crit, 
        psi_wilt=psi_wilt, psi_crit=psi_crit)
                                                            
    a1 = psi_wilt / RHO_W / G_GRAVITY / 0.01
    a2 = (0.505 - (0.037 * (b - 3.1) / 15.7)) / sm_wilt
    a3 = (-0.142 - 0.037 * 0.3 / 15.7) / sm_wilt
    a4 = 2.17 - 0.63 * (b - 3.1) / 15.7
    a5 = -0.63 / 15.7 * 0.3 - 1.58
    
    # in Wolfram Alpha
    # Solve[a1 *(a2 + a3 x)^(-b) == 10^(a4 + a5 x), {x}]
    
    def productlog(x):
        if x < 0.0:
            raise ValueError('x needs to be >= 0, but instead is' + str(x))
        y = float(scipy.special.lambertw(x))
        
        return y
        
    f_sand = ((-(a2*a5*math.log(10.0)) + a3*b*productlog(
             a5/(a3*b*(10.0**(a4 - (a2*a5)/a3)/(a1*math.log(10.0)**b))**b**(-1.0))))
             /(a3*a5*math.log(10.0)))
    
    f_clay = (b - 3.1 + 0.3 * f_sand) / 15.7
    
    f_silt = 1.0 - f_clay - f_sand
    
    if not np.all(np.array([f_clay, f_sand, f_silt]) > 0.0):
        print(f_clay, f_sand, f_silt)
        raise Exception('one of the fractions went below zero')
    
    return f_clay, f_sand, f_silt
 

def calc_b_from_sm_wilt_sm_crit(sm_wilt=None, sm_crit=None, psi_wilt=1.5E6, psi_crit=0.033E6):

    b = math.log(psi_crit / psi_wilt) / math.log(sm_wilt / sm_crit) # math.log is to log to base e
    
    return b
 

def sathh_from_sm_wilt_sm_crit_sm_sat(sm_wilt=None, sm_sat=None, sm_crit=None, psi_wilt=1.5E6, psi_crit=0.033E6):

    b = calc_b_from_sm_wilt_sm_crit(sm_wilt=sm_wilt, sm_crit=sm_crit, psi_wilt=psi_wilt, psi_crit=psi_crit)
    
    sathh = ( sm_wilt / sm_sat ) ** b * psi_wilt / (RHO_W * G_GRAVITY)
    
    # checking consistency:
    # another_sathh = ( sm_crit / sm_sat ) ** b * psi_crit / (RHO_W * G_GRAVITY) 
    # another_sm_wilt = sm_sat * ( sathh * RHO_W * G_GRAVITY / psi_wilt ) ** (1.0 / b)    
    # another_sm_crit = sm_sat * ( sathh * RHO_W * G_GRAVITY / psi_crit ) ** (1.0 / b)
    
    return sathh


def load_from_jules_ancil(filename):
    '''
    Loads the soil properties from a JULES soil ancil text
    file, assume format generated by the print_jules_soil_parameters function
    (i.e. headings are JULES var names, file only contains one soil type.)
    '''

    data = np.genfromtxt(filename, names=True)
    soil = SoilType(empty=True)
    for var in data.dtype.names:
        soil.jules_soil_parameters[var] = data[var]

    return soil


def run_from_command_line():
    '''
    Allows an instance of the SoilType class to be created from the command line and
    the parameters to be written to standard out.
    '''

    epilog = __version__ + """

Example 1:
    ./soil_cosby_parameters.py --soil_textural_class='Silty clay loam'

Example 2:
    ./soil_cosby_parameters.py --f_sand=0.10 --f_silt=0.56 --f_clay=0.34

Example 3:
    ./soil_cosby_parameters.py --soil_textural_class='Silty clay loam' --f_organic=0.8 --soc_layer=1

Example 4
    ./soil_cosby_parameters.py --f_sand=0.10 --f_silt=0.26 --f_clay=0.64 --l_cap_cutoffs

-----------------------
    """

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=epilog)

    for opt in ["f_sand", "f_clay", "f_silt"]:
        parser.add_argument(
            '--' + opt, action='store', dest=opt, type=float,
            help='fraction of ' + opt[2:] + ' in the mineral soil component')

    parser.add_argument(
        '--f_organic', action='store', dest='f_organic',
        type=float, help='organic fraction of soil')

    parser.add_argument(
        '--soil_textural_class', action='store',
        dest='soil_textural_class',
        help='soil textural class i.e. one of ' +
        str(list(SOIL_TEXTURAL_CLASS.keys())),
        type=str)

    parser.add_argument(
        '--soc_layer', action='store', dest='soc_layer',
        help='layer to use for the organic soil properties (see Dankers 2011)',
        type=int)

    parser.add_argument(
        '--l_cap_cutoffs', action='store_true', dest='l_cap_cutoffs',
        help='Apply CAP cutoffs for very fine or organic soil. Cannot be used with the Dankers 2011 organic soil parameterisation.')

    parser.set_defaults(l_cap_cutoffs=False)

    parser.add_argument('--version', action='version', version=__version__)

    # vars converts the Namespace object to a dictionary
    kwargs = vars(parser.parse_args())

    try:
        soil = SoilType(**kwargs)
    except:
        parser.print_help()
        raise

    soil.print_jules_soil_parameters()
    
    print('\nwhere\n')
    for key,val in COSBY_PARAMETER_DESCRIPTION.iteritems():
        print(key + ' is the ' + val)

    return


if __name__ == '__main__':

    run_from_command_line()
