# -*- coding: iso-8859-1 -*-

'''

Python module using the model of stomatal conductance from Huntingford et al 2015. 
(n.b. this is not the code used in the Huntingford et al 2015 paper - it was written retrospectively.)
Note that Huntingford et al 2015 considers root synthesised ABA only and assumes 
root-leaf epidermis hydraulic resistance is constant, whereas this code has been extended to
allow these factors to be taken into account.

Karina Williams and Chris Huntingford.


Notes:

 1) Huntingford et al 2015 use table 2 in Cox et al 1999. This table is incorrect because it has been 
   calculated assuming that the logarithms in Cosby et al 1984 are to the base e when they
   are actually to the base 10.
 
 2) The value of a in Huntingford et al 2015 should be 1.4E-3 mol ABA mg m-5 s-1 MPa-1
   rather than -1.4E-3 mol ABA mg m-5 s-1 MPa-1 (see Tardieu et al 1993).
 
 3) The value of R_p in Huntingford et al 2015 is correct. It cites Tardieu et al 1993, which 
   cites Saugier 1991. The value in Tardieu et al 1993 is out by a factor 1E3.
   
 4) Ksat in Huntingford et al 2015 is out by a factor 1E3, if the starting point is Ksat from
   table 2 in Cox et al 1999. (this means the 10^-6 in R_ROOT aba.f should be 1-^-9)
   
 5) KAPPA in Huntingford et al 2015 should be 2.34E7 * (0.27/22.4) m3 ppm (mol C02)-1
   This is because equation 14 is written in a different form in aba.f (directly involving
   the gs calculated with equation 1).

 6) Huntingford et al 2015 gives delta = -0.183 MPa-1 and beta = -2.69E3 m3 mol-1 (with Psi_l is in
    MPa and c_ABA is in mol ABA m3 water-1).
    It cites Tardieu and Davies 1993 which gives delta = 0.183 and beta = -2.69 (no units  
    are given).
    Comparing with Tardieu and Davies 1993 fig 1, I think it should be delta = -2.69 MPa-1 and 
    beta = -0.183E3 m3 water (mol ABA)-1 (with Psi_l is in
    MPa and c_ABA is in mol ABA m3 water-1).
   
 7) In Huntingford et al 2015, half mean distance between roots in m (d) is 5.6E-3 m
    mean radius of the roots in m (r) is 5.0E-4 m and root length per unit area of ground in m m-2
    (L_a) is 1.0E4 m-1. Huntingford et al 2015 says these "are similar to those Tardieu and Davies 1993 
    and correspond to typical for maize (Zea Mays L)".
    Tardieu and Davies 1993 cites Tardieu 1991 for these parameters, which uses
    d = 17.4E-3 m, r = 0.8E-3 m and L_a = 1050.0 * 0.4 m-1 and only models the top 0.4m of soil. These
    are the values we use here.


"aba.f" refers to the fortran code that Chris used to produce Fig 1 in Huntingford et al 2015 (see email from CH 1.2.16).


References:

 * Huntingford et al 2015:
       Combining the [ABA] and net photosynthesis-based model equations of stomatal conductance
       Chris Huntingford, D. Mark Smith, William J. Davies, Richard Falk, Stephen Sitch, Lina M. Mercado
       Ecological Modelling, Vol. 300 (March 2015), pp. 81-88
       http://dx.doi.org/10.1016/j.ecolmodel.2015.01.005

 * Cox et al 1999
       The impact of new land surface physics on the GCM simulation of climate and climate sensitivity 
       P. M. Cox, R. A. Betts, C. B. Bunton, R. L. H. Essery, P. R. Rowntree, J. Smith
       Climate Dynamics In Climate Dynamics, Vol. 15, No. 3. (8 March 1999), pp. 183-203,
       http://dx.doi.org/10.1007/s003820050276
      
 * Cosby et al 1984      
      A Statistical Exploration of the Relationships of Soil Moisture Characteristics to the Physical Properties of Soils
      B. J. Cosby, G. M. Hornberger, R. B. Clapp, T. R. Ginn
      Water Resour. Res., Vol. 20, No. 6. (1 June 1984), pp. 682-690
      http://dx.doi.org/10.1029/wr020i006p00682
   
 * Tardieu et al 1993   
      Integration of hydraulic and chemical signalling in the control of stomatal conductance and water status of droughted plants
      F. Tardieu, W. J. Davies
      Plant, Cell & Environment, Vol. 16, No. 4. (1 May 1993), pp. 341-349
      http://dx.doi.org/10.1111/j.1365-3040.1993.tb00880.x
      
 * Saugier 1991      
      Some plant factors controlling evapotranspiration
      Bernard Saugier, Nader Katerji
      Agricultural and Forest Meteorology, Vol. 54, No. 2-4. (April 1991), pp. 263-277
      http://dx.doi.org/10.1016/0168-1923(91)90009-f
      
 * Verhoef and Egea 2014
      Modeling plant transpiration under limited soil water: Comparison of different plant and soil hydraulic parameterizations and preliminary implications for their use in land surface models
      Anne Verhoef, Gregorio Egea
      Agricultural and Forest Meteorology, Vol. 191 (June 2014), pp. 22-32
      http://dx.doi.org/10.1016/j.agrformet.2014.02.009
      
'''


import numpy as np
import soil_cosby_parameters


# Global constants. See notes section for more information on how these differ from those used
# in Huntingford 2015
# ToDo: define a new object instead of using lots of global variables     
    
# minimum stomatal conductance in m s-1
GS_MIN = 8.93E-4 # corresponds to 0.02 mol m-2 s-1 in physiological units
    
# ALPHA is (gs_max-GS_MIN) in m s-1
ALPHA = 0.0112
    
# DELTA in MPa-1
DELTA = -2.69 # not the same as used in Huntingford 2015 paper

# BETA in m3 mol-1
BETA = -0.183E3 # not the same as used in Huntingford 2015 paper

# KAPPA in m3 ppm (mol CO2)^{-1}
KAPPA_PAPER = 2.34E7 # value in Huntingford 2015 paper
KAPPA = KAPPA_PAPER * 0.27 / 22.4 # changed after phone conversation with CH 2.2.16

# half mean distance between roots in m
D = 17.4E-3 # used in Tardieu 1991 for top 0.4m of soil for maize
# not the same as used in Huntingford 2015 paper

# mean radius of the roots in m 
R = 0.8E-3 # used in Tardieu 1991 for top 0.4m of soil for maize
# not the same as used in Huntingford 2015 paper

# root length per unit area of ground in m m-2
L_A = 1050.0 * 0.4 # used in Tardieu 1991 for top 0.4m of soil for maize
# not the same as used in Huntingford 2015 paper

# plant resistance which includes both radial resistance 
# within the roots and axial resistance between the roots and leaves
# Unit: MPa m2 s mg-1
R_P_MIN = 7.2E-3

# Coefficient of water potential in root in c_ABA calculation. Unit: mol ABA mg m-5 s-1 MPa-1
AR_ABA = 1.4E-3 # in aba.f (AR_ABA = -1.4E-3 in paper)

# Coefficient of water potential in leaf in c_ABA calculation. Unit: mol ABA mg m-5 s-1 MPa-1
AL_ABA = 0.0

# Constant in denominator of c_ABA calculation. Unit: mg m-2 s-1
B_ABA = 4.0

# threshold value of xylem hydraulic conductivity at which leaf potential starts to decline
PSI_TL = -998.0 # MPa. Setting very small to switch off

# value of xylem hydraulic conductivity at which leaf potential falls to zero
PSI_XL = -999.0 # MPa. Setting very small to switch off

MOLAR_MASS_WATER_IN_G_PER_MOL = 18.0153


def calc_gs_section_2p3_method(soil_water_percent, soil, J_w=None, a_n=None, c3=True, c_ast=None, Psi_s=None, K=None):
    '''
    gs is the stomatal conductance in m s-1
    J_w is evapotranspiration in mg m-2 s-1
    a_n is net assimulation mol CO2 m-2 s-1 
    Uses equation 14 in section 2.14 
    '''
    
    c_ABA = calc_c_ABA(soil_water_percent, soil, J_w=J_w, Psi_s=Psi_s, K=K)
    
    Psi_l = calc_Psi_l(soil_water_percent, soil, J_w=J_w, Psi_s=Psi_s, K=K)

    # atmospheric CO2 concentration in ppm
    c_a = 350.0

    # temperature in degrees C
    tdegc = 303.15 - 273.15

    # pressure in Pa
    pstar = 1.013E5

    if c_ast is None:
        c_ast = calc_c_ast(c3, tdegc=tdegc, pstar=pstar)

    f = calc_f_of_D()

    gs_ABA = calc_gs_section_2p1_method(soil_water_percent, soil, J_w=J_w, Psi_s=Psi_s, K=K)
    
    # leaf level stomatal conductance in m s-1, equation 14
    gs = GS_MIN + ( KAPPA * f * a_n ) / ( c_a - c_ast ) * ( np.exp( c_ABA * BETA * np.exp( DELTA * Psi_l ) ) )
    #to be a closer match to aba.f, would want
    #gs = GS_MIN + ( KAPPA_PAPER * f * a_n ) / ( c_a - c_ast ) * (GS_MIN + ALPHA * np.exp( c_ABA * BETA * np.exp( DELTA * Psi_l ) ) )
       
    gs = np.maximum(GS_MIN, gs)
    
    return gs


def calc_gs_section_2p1_method(soil_water_percent, soil, J_w=None, Psi_s=None, K=None):
    '''
    gs is the stomatal conductance in m s-1
    J_w is evapotranspiration in mg m-2 s-1
    Uses equation 1 in section 2.1 (which is not the final method recommended by the paper)
    '''
    
    c_ABA = calc_c_ABA(soil_water_percent, soil, J_w=J_w, Psi_s=Psi_s, K=K)
    
    Psi_l = calc_Psi_l(soil_water_percent, soil, J_w=J_w, Psi_s=Psi_s, K=K)

    # leaf level stomatal conductance in m s-1, equation 1
    gs = GS_MIN + ALPHA * np.exp( c_ABA * BETA * np.exp( DELTA * Psi_l ) )

    gs = np.maximum(GS_MIN, gs)
    
    return gs


def calc_c_ast(c3, tdegc=None, pstar=None):
    '''
    c_ast is the temperature- and atmospheric oxygen-dependent CO2 compensation point
    in ppm
    
    n.b. aba.f specifies c_ast directly, rather than calculating it as we do here
    '''

    if c3:
       power = 0.1 * (tdegc - 25.0) # jules sf_stom r3052
       tau = 2600.0 * (0.57 ** power) # jules sf_stom r3052

       # Ratio of molecular weights of O2 and dry air, jules ccarbon.h r3052
       epo2 = 1.106

       # Atmospheric concentration of oxygen (kg O2/kg air), jules jules_surface_mod.F90 r3052
       o2 = 0.23

       oa = o2 / epo2 * pstar # Atmospheric O2 pressure in Pa, jules leaf_limits r3052
       # n.b. in Huntingford et al 2015, oa is set directly to 21031.0

       c_ast = 0.5 * oa / tau  # jules leaf_limits r3052
       # code says mol/m3, but assume it's actually Pa (it gets subtracted from canopy CO2 pressure
       # in Pa later)
       # reference in code is HCTN 24 eq 53 also says unit is Pa       
       
       c_ast = c_ast / pstar # ratio of co2 particles to air particles
       c_ast *= 1E6 # ppm

    else:
       # for C4 plants
       c_ast = 0.0

    return c_ast


def calc_f_of_D():
    '''
    function of humidity deficit 
    '''

    # D_0 in kg kg-1
    D_0 = 0.09
   
    # D in kg kg-1
    D = 0.008

    #f = 1.0 / (1.0 + D / D_0 )

    f = 1.0

    return f


def calc_c_ABA(soil_water_percent, soil, J_w=None, Psi_s=None, K=None):
    '''
    c_ABA is the ABA concentration in mol [ABA] (m3 water)-1
    J_w is evapotranspiration in mg m-2 s-1
    '''
    
    Psi_r = calc_Psi_r(soil_water_percent, soil, J_w=J_w, Psi_s=Psi_s, K=K)
    
    if np.abs(AL_ABA) > 0.0:
        # There is a both a root water potential term and a
        # leaf water potential term, as in Verhoef and Egea 2014 equation 6.
        Psi_l = calc_Psi_l(soil_water_percent, soil, J_w=J_w, Psi_s=Psi_s, K=K)
        c_ABA = - ( AR_ABA * Psi_r + AL_ABA * Psi_l ) / ( J_w + B_ABA )
    else: 
        # ABA concentration just depends on potential
        # in root, as in Huntingford et al 2015 equation 2.
        c_ABA = - AR_ABA * Psi_r / ( J_w + B_ABA ) 
    
    return c_ABA
    
    
def calc_Psi_l(soil_water_percent, soil, J_w=None, Psi_s=None, K=None):
    '''
    Calculates leaf water potential by solving eqn 7 and eqn 8
    in Verhoef and Egea 2014 as simultaneous equations (but note different units)
    and always taking the highest root (i.e. nearest to zero).
    In Huntingford et al 2015, Psi_l was always calculated with R_P_MIN -
    to reproduce this, make sure PSI_TL is set to a very low value
    so that Psi_l_initial is always greater than PSI_TL.

    Args:
      * soil_water_percent: volumetric soil water as a percentage
      * soil: soil_cosby_parameters.SoilType object specifying the soil type.

    Kwargs:
      * J_w: soil to leaf water flux in mg m-2 s-1 (not mol m-2 s-1, as in Verhoef and Egea 2014)

    returns
      * Psi_l: water potential of the bulk leaf epidermis in MPa
           
    '''
    
    # R_sp is in MPa m2 s mg-1 (n.b. different units to R_sr in Verhoef and Egea 2014)
    R_sp = calc_R_sp(soil_water_percent, soil, K=K)

    Psi_s = calc_Psi_s(soil_water_percent, soil)
    Psi_r = calc_Psi_r(soil_water_percent, soil, J_w=J_w, Psi_s=Psi_s, K=K)
    
    # leaf water potential in MPa using minimum R_p
    Psi_l_initial = Psi_r - J_w * R_P_MIN 

    if Psi_l_initial >= PSI_TL:
        Psi_l = Psi_l_initial
    else:
        a = 1.0
        b = - PSI_XL - Psi_s + J_w * R_sp
        c = PSI_XL * (Psi_s - J_w * R_sp) + J_w * R_P_MIN * (PSI_TL - PSI_XL) 

        discriminant = b ** 2.0 - 4.0 * a * c
        if discriminant >= 0.0:
            Psi_l = ( -b + np.sqrt(discriminant) ) / (2.0 * a)
        else:
            Psi_l = np.nan
            
    return Psi_l


def calc_Psi_r(soil_water_percent, soil, J_w=None, Psi_s=None, K=None):
    '''
    Psi_r is the water potential at the root surface in MPa
    J_w is evapotranspiration in mg m-2 s-1
    '''    
    
    R_sp = calc_R_sp(soil_water_percent, soil, K=K)
    if Psi_s is None:
        Psi_s = calc_Psi_s(soil_water_percent, soil)
    
    #Psi_r is the water potential at the root surface in MPa, equation 4
    Psi_r = Psi_s - J_w * R_sp
    
    return Psi_r


def calc_Psi_s(soil_water_percent, soil):
    '''
    Psi_s is the bulk soil water potential in MPa
    '''

    Psi_sat = soil.jules_soil_parameters['sathh'] # in m
    Psi_sat = convert_m_water_to_Pa(Psi_sat)
    Psi_sat *= 1.0E-6 # in MPa
    
    # Psi_s is the bulk soil water potential in MPa, equation 6
    Psi_s = Psi_sat * (soil_water_percent / 100.0 / soil.jules_soil_parameters['sm_sat']) ** \
           (-soil.jules_soil_parameters['b'])
    
    return Psi_s
    
    
def convert_Pa_to_m_water(val):
    rho_w = 999.97  # density of water kg m-3 (at 4 degC)
    g = 9.81  # acceleration due to gravity in m s-2
 
    return - val / (rho_w * g)
    

def convert_m_water_to_Pa(val):
    conv = convert_Pa_to_m_water(1.0)
    
    return val / conv


def convert_mg_per_m_per_s_per_MPa_to_kg_per_m2_per_s(val):
    '''
    Unit conversion from units used in Huntingford et al 2015 Ksat to JULES Ksat
    '''
    new_val = val # mg_per_m_per_s_per_MPa
    new_val *= 1.0E-6 # mg_per_m_per_s_per_Pa
    new_val *= 1.0E-6 # kg_per_m_per_s_per_Pa
    new_val = convert_m_water_to_Pa(new_val) # kg_per_m2_per_s
    new_val = np.abs(new_val)

    return new_val 


def convert_kg_per_m2_per_s_to_mg_per_m_per_s_per_MPa(val):
    conv = convert_mg_per_m_per_s_per_MPa_to_kg_per_m2_per_s(1.0)

    return val / conv 


#def convert_kg_C_per_m2_per_s_to_mol_CO2_per_m2_per_s(val):
    #print 'have not checked this yet'

    #val *= 1.0E3 # g C m-2 s-1
    #val /= 0.272912 # g CO2 m-2 s-1 (CO2 is 27% carbon by mass)  
    #val /= 44.0095 # mol CO2 m-2 s-1 ( a mol of CO2 is 44.0095 g )

    #return val


def convert_W_per_m2_to_mg_water_per_m2_per_s(val):

    latent_heat_of_evap_of_water = 2.501E6 # at 0degC in J kg-1, from JULES water_constants_mod r3052
    val = val / latent_heat_of_evap_of_water # kg m-2 s-1 
    val *= 1.0E6 # mg m-2 s-1

    return val


def calc_R_sp(soil_water_percent, soil, K=None):
    '''
    R_sp is the resistance between the soil and the root surface in MPa m2 s mg-1
    '''

    if K is None:
        K = calc_soil_hydraulic_conductivity(soil_water_percent, soil)    

    # R_sp is the resistance between the soil and the root surface in MPa m2 s mg-1, equation 5
    R_sp = 1.0 / (4.0 * np.pi * K * L_A ) * np.log( D**2.0 / R**2.0 )

    return R_sp


def calc_soil_hydraulic_conductivity(soil_water_percent, soil):
    ''' 
    soil hydraulic conductivity in mg m-1 s-1 MPa-1, equation 6
    '''

    # saturated soil hydraulic conductivity in mg m-1 s-1 MPa-1
    K_sat = np.abs(convert_kg_per_m2_per_s_to_mg_per_m_per_s_per_MPa(soil.jules_soil_parameters['satcon'])) # satcon is in kg m-2 s-1

    # soil hydraulic conductivity in mg m-1 s-1 MPa-1, equation 6
    K = K_sat * (soil_water_percent / 100.0 / soil.jules_soil_parameters['sm_sat']) ** \
           ( 2.0 * soil.jules_soil_parameters['b'] + 3.0 )

    return K


def fill_soil_with_wrong_parameters(soil):
    # see http://www-nwp/~frsurf/ANCIL/view/dev/doc/AncilDoc_Cosby.html

    # this has 10** -> exp
    soil.jules_soil_parameters['sathh'] = \
        0.01 * (np.exp(2.17 - 0.63 * soil.f_clay - 1.58 * soil.f_sand))        
        
    # this has 10** -> exp and 2.75 -> 5.55 (because Ks in Cosby is in inches per hour)
    soil.jules_soil_parameters['satcon'] = \
        np.exp(-5.55 - 0.64 * soil.f_clay + 1.26 * soil.f_sand)
        
    soil.jules_soil_parameters['sm_wilt'] = soil._theta(psi=1.5E6)
    soil.jules_soil_parameters['sm_crit'] = soil._theta(psi=0.033E6)

    return soil 


def main():
    pass 


if __name__ == '__main__':
    main()
