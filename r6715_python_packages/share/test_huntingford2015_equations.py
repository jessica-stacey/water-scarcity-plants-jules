# -*- coding: iso-8859-1 -*-

import unittest
import mock

import numpy as np

import soil_cosby_parameters
import huntingford2015_equations


class TestUnitConversions(unittest.TestCase):

    def test_convert_Pa_to_m(self):
        psisat_from_cox1999 = 4.95E-2
        psisat_from_huntingford2015 = -4.86E-4 * 1E6 # Pa 
        test = huntingford2015_equations.convert_Pa_to_m_water(psisat_from_huntingford2015)
        np.testing.assert_allclose(test, psisat_from_cox1999, rtol=5e-03)
    
    def test_convert_m_to_Pa_and_convert_Pa_to_m_are_inverse(self):
        a = 1.0
        a = huntingford2015_equations.convert_Pa_to_m_water(a)
        a = huntingford2015_equations.convert_m_water_to_Pa(a)
        np.testing.assert_allclose(a, 1.0, rtol=5e-03)
    
    def test_convert_mg_per_m_per_s_per_MPa_to_kg_per_m2_per_s(self):
        Ksat_from_cox1999 = 4.72E-3 #mm s-1 water(=kg m-2 s-1 water)

        Ksat_from_huntingford2015 = 4.81E2 # mg m-1 s-1 MPa-1
        Ksat_from_huntingford2015 *= 1.0E3 # email conversation with CH 1.2.16

        test = huntingford2015_equations.convert_mg_per_m_per_s_per_MPa_to_kg_per_m2_per_s(Ksat_from_huntingford2015)
        np.testing.assert_allclose(test, Ksat_from_cox1999, rtol=5e-03)
        
        
@mock.patch('huntingford2015_equations.BETA', -1.48E2)
@mock.patch('huntingford2015_equations.DELTA', -2.0)
@mock.patch('huntingford2015_equations.B_ABA', 1.8)
@mock.patch('huntingford2015_equations.AR_ABA', 4.0E-3)
@mock.patch('huntingford2015_equations.AL_ABA', 1.0E-3)
@mock.patch('huntingford2015_equations.D', 18.3058E-3)
@mock.patch('huntingford2015_equations.R', 0.8E-3)
@mock.patch('huntingford2015_equations.L_A', 299.0)
@mock.patch('huntingford2015_equations.R_P_MIN', 13.889E-3)
@mock.patch('huntingford2015_equations.PSI_TL', -1.0)
@mock.patch('huntingford2015_equations.PSI_XL', -7.0)
class TestAgainstAVCode(unittest.TestCase):
    '''
    All KGO generated with an edited version of Betacomparisonallpeachnewenbal.r
    and MeteoAvignonuntil2014loam.csv
    (changes the second 'if' in the Lre calculation to 'else if')
    See plot.peach.py for discussion of all the parameters.
    '''
    def setUp(self):
        # KGO copied from output file
        
        #year    month   day     hour    mins
        #2001    4       25      12      5
        #2002  6       17      15      0
        #2007    6       3       8       0
        #2009  9       9       13      0
        #2014    6       16      9       0

        self.soil_water_percent = np.array([0.296899989989895, 0.164499995190311, 
            0.189099999349167, 0.238800012212961, 0.155300005010254]) * 100.0
        self.El = np.array([0.755364737915755, 2.88115540124775, 1.98075655258495, 
            6.49917470630336, 0.56990395143449]) 
        self.psi_l = [-0.234397888183594, -2.11417388916016, -1.02854156494141, 
            -2.16533660888672, -1.71598052978516]
        self.psi_r = [-0.0455163527473942, -1.22962123573173, -0.531028537835749, 
            -0.148931163179201, -1.55419267030133]
        self.psi_s = [-0.0455062, -1.0972493, -0.5176928, 
            -0.1471751, -1.4963226]
        self.beta_eff = [0.993622924427568, 0.26429859344356, 0.907139047582368, 
            0.769946207890612, 0.0491838846962456]
        self.c_ABA = np.array([27.0491042341095, 131.057665858598, 84.1749312652603, 
            23.2441629655282, 657.86804163026]) * 1.0E-6
        self.R_re = [250.0, 307.010516946572, 251.194916001334, 
            310.259449035726, 283.874805620088]
        
        self.soil = self.fill_soil()
        self.Jw = self.El * huntingford2015_equations.MOLAR_MASS_WATER_IN_G_PER_MOL
        self.npoints = len(self.El)
    
    def fill_soil(self):

        # first filling the soil object with Cosby parameters
        soil = soil_cosby_parameters.SoilType(soil_textural_class='Loam')  
        
        # Now replacing with soil parameters from Anne for loam, see email 20.6.2017
        soil.jules_soil_parameters['b'] = 5.39
        soil.jules_soil_parameters['sm_sat'] = 0.451
        soil.jules_soil_parameters['sathh'] = 0.00478E6 / ( soil_cosby_parameters.RHO_W * soil_cosby_parameters.G_GRAVITY ) 
        soil.jules_soil_parameters['satcon'] = 0.0417E-2 / 60 * 1E3  
        soil.jules_soil_parameters['sm_wilt'] = soil_cosby_parameters.brooks_and_corey_equation(soil, psi=1.5E6)
        soil.jules_soil_parameters['sm_crit'] = soil_cosby_parameters.brooks_and_corey_equation(soil, psi=0.033E6)
        
        return soil
        
    def test_calc_Psi_s(self):
        
        for i in range(self.npoints):
            test = huntingford2015_equations.calc_Psi_s(self.soil_water_percent[i], self.soil)
            np.testing.assert_allclose(test, self.psi_s[i], rtol=1e-6)
            
    def test_calc_Psi_r(self):
        
        for i in range(self.npoints):
            test = huntingford2015_equations.calc_Psi_r(self.soil_water_percent[i], self.soil, J_w=self.Jw[i])
            np.testing.assert_allclose(test, self.psi_r[i], rtol=5e-04)
            
    def test_calc_Psi_l(self):
        
        for i in range(self.npoints):
            test = huntingford2015_equations.calc_Psi_l(self.soil_water_percent[i], self.soil, J_w=self.Jw[i])
            np.testing.assert_allclose(test, self.psi_l[i], rtol=5e-03)

    def test_calc_c_ABA(self):
        
        for i in range(self.npoints):
            test = huntingford2015_equations.calc_c_ABA(self.soil_water_percent[i], self.soil, J_w=self.Jw[i])
            np.testing.assert_allclose(test, self.c_ABA[i], rtol=1e-03)

    def test_calc_beta_eff(self):
        
        for i in range(self.npoints):
            gs = huntingford2015_equations.calc_gs_section_2p1_method(self.soil_water_percent[i], self.soil, J_w=self.Jw[i])
            test = ( gs - huntingford2015_equations.GS_MIN ) / huntingford2015_equations.ALPHA
            np.testing.assert_allclose(test, self.beta_eff[i], rtol=5e-03)
            
    def test_compare_R_re(self):
        
        for i in range(self.npoints):
            psi_l = huntingford2015_equations.calc_Psi_l(self.soil_water_percent[i], self.soil, J_w=self.Jw[i])
            psi_s = huntingford2015_equations.calc_Psi_s(self.soil_water_percent[i], self.soil)
            R_sp = huntingford2015_equations.calc_R_sp(self.soil_water_percent[i], self.soil)
            
            R_re = (psi_s - psi_l) / self.Jw[i] - R_sp # in MPa m2 s mg-1
            R_re *= 1.0E3 # in MPa m2 s g-1  
            R_re *= huntingford2015_equations.MOLAR_MASS_WATER_IN_G_PER_MOL # in mol-1 m2 s MPa
            
            np.testing.assert_allclose(R_re, self.R_re[i], rtol=5e-03)


@mock.patch('huntingford2015_equations.BETA', -2.69E3)
@mock.patch('huntingford2015_equations.DELTA', -0.183)
@mock.patch('huntingford2015_equations.R', 5.0E-4)
@mock.patch('huntingford2015_equations.L_A', 1.0E4)
@mock.patch('huntingford2015_equations.D', 5.6E-3)
class TestPaperEquationAgainstAbaDotF(unittest.TestCase):
    '''
    All kgo generated using an edited version of aba.f 
    '''       
    def test_calc_soil_hydraulic_conductivity(self):
        
        soil = soil_cosby_parameters.SoilType(f_sand=0.27, f_silt=0.5, f_clay=0.23)
        soil = huntingford2015_equations.fill_soil_with_wrong_parameters(soil) 
        soil_water_percent = 13.5

        test = huntingford2015_equations.calc_soil_hydraulic_conductivity(soil_water_percent, soil)
        kgo = 1.13717595E-03

        np.testing.assert_allclose(test, kgo, rtol=5e-02)

    def test_calc_R_sp(self):
        
        soil = soil_cosby_parameters.SoilType(f_sand=0.27, f_silt=0.5, f_clay=0.23)
        soil = huntingford2015_equations.fill_soil_with_wrong_parameters(soil) 
        soil_water_percent = 13.5

        test = huntingford2015_equations.calc_R_sp(soil_water_percent, soil)
        kgo = 3.38122658E-02

        np.testing.assert_allclose(test, kgo, rtol=5e-02)

    def test_calc_Psi_s(self):
        
        soil = soil_cosby_parameters.SoilType(f_sand=0.27, f_silt=0.5, f_clay=0.23)
        soil = huntingford2015_equations.fill_soil_with_wrong_parameters(soil) 
        soil_water_percent = 13.5

        test = huntingford2015_equations.calc_Psi_s(soil_water_percent, soil)
        kgo = -1.5984513

        np.testing.assert_allclose(test, kgo, rtol=5e-03)

    def test_calc_Psi_r(self):
        
        soil = soil_cosby_parameters.SoilType(f_sand=0.27, f_silt=0.5, f_clay=0.23)
        soil = huntingford2015_equations.fill_soil_with_wrong_parameters(soil) 
        soil_water_percent = 13.5
        J_w = 130.46968

        test = huntingford2015_equations.calc_Psi_r(soil_water_percent, soil, J_w=J_w)
        kgo = -6.0099268

        np.testing.assert_allclose(test, kgo, rtol=1e-02)

    def test_calc_Psi_l(self):
        
        soil = soil_cosby_parameters.SoilType(f_sand=0.27, f_silt=0.5, f_clay=0.23)
        soil = huntingford2015_equations.fill_soil_with_wrong_parameters(soil) 
        soil_water_percent = 13.5
        J_w = 130.46968

        test = huntingford2015_equations.calc_Psi_l(soil_water_percent, soil, J_w=J_w)
        kgo = -6.9493084

        np.testing.assert_allclose(test, kgo, rtol=1e-02)

    def test_calc_c_ABA(self):
        
        soil = soil_cosby_parameters.SoilType(f_sand=0.27, f_silt=0.5, f_clay=0.23)
        soil = huntingford2015_equations.fill_soil_with_wrong_parameters(soil) 
        soil_water_percent = 13.5
        J_w = 130.46968

        test = huntingford2015_equations.calc_c_ABA(soil_water_percent, soil, J_w=J_w)
        kgo = 6.25709581E-05

        np.testing.assert_allclose(test, kgo, rtol=5e-02)

    def test_calc_gs_section_2p1_method(self):
        
        soil = soil_cosby_parameters.SoilType(f_sand=0.27, f_silt=0.5, f_clay=0.23)
        soil = huntingford2015_equations.fill_soil_with_wrong_parameters(soil) 
        soil_water_percent = 13.5
        J_w = 130.46968

        test = huntingford2015_equations.calc_gs_section_2p1_method(soil_water_percent, soil, J_w=J_w)
        kgo = 7.01564923E-03

        np.testing.assert_allclose(test, kgo, rtol=5e-03)

    def test_calc_c_ast(self):
        
        c3 = True
        tdegc = 303.15 - 273.15
        pstar = 1.013E5

        test = huntingford2015_equations.calc_c_ast(c3, tdegc=tdegc, pstar=pstar)
        kgo = 52.882397

        np.testing.assert_allclose(test, kgo, rtol=5e-03)

    def test_calc_gs_section_2p3_method(self):
        
        soil = soil_cosby_parameters.SoilType(f_sand=0.27, f_silt=0.5, f_clay=0.23)
        soil = huntingford2015_equations.fill_soil_with_wrong_parameters(soil) 
        soil_water_percent = 13.5
        J_w = 130.46968
        c3 = True
        a_n = 1.08239192E-05 
        c_ast = 52.882397

        test = huntingford2015_equations.calc_gs_section_2p3_method(soil_water_percent, soil, J_w=J_w, a_n=a_n, c3=c3, c_ast=c_ast)
        kgo = 6.87353220E-03

        # really loose condition because implementation is different
        np.testing.assert_allclose(test, kgo, rtol=1e-01)

@mock.patch('huntingford2015_equations.R', 0.8E-3)
@mock.patch('huntingford2015_equations.L_A', 299.0)
@mock.patch('huntingford2015_equations.D', 18.3058E-3)
@mock.patch('huntingford2015_equations.R_P_MIN', 13.889E-3)
@mock.patch('huntingford2015_equations.PSI_TL', -1.0)
@mock.patch('huntingford2015_equations.PSI_XL', -7.0)
class TestCalcPsiL(unittest.TestCase):

    def test_examples(self):
        soil = soil_cosby_parameters.SoilType(soil_textural_class='Loam')  
        soil.jules_soil_parameters['b'] = 5.39
        soil.jules_soil_parameters['sm_sat'] = 0.451
        soil.jules_soil_parameters['sathh'] = 0.00478E6 \
            / ( soil_cosby_parameters.RHO_W * soil_cosby_parameters.G_GRAVITY ) 
        soil.jules_soil_parameters['satcon'] = 0.0417E-2 / 60 * 1E3  
        
        soil_water_percent = 16.0
        El = 1.0
        psi_l_kgo = -1.62069891418
        J_w = El * huntingford2015_equations.MOLAR_MASS_WATER_IN_G_PER_MOL
        psi_l = huntingford2015_equations.calc_Psi_l(soil_water_percent, soil, J_w=J_w)
        np.testing.assert_allclose(psi_l, psi_l_kgo)
        
        soil_water_percent = 25.0
        El = 1.0
        psi_l_kgo = -0.365317335945
        J_w = El * huntingford2015_equations.MOLAR_MASS_WATER_IN_G_PER_MOL
        psi_l = huntingford2015_equations.calc_Psi_l(soil_water_percent, soil, J_w=J_w)
        np.testing.assert_allclose(psi_l, psi_l_kgo)
        
        soil_water_percent = 25.0
        El = 4.0
        psi_l_kgo = -1.14036814274
        J_w = El * huntingford2015_equations.MOLAR_MASS_WATER_IN_G_PER_MOL
        psi_l = huntingford2015_equations.calc_Psi_l(soil_water_percent, soil, J_w=J_w)
        np.testing.assert_allclose(psi_l, psi_l_kgo)
        
        soil_water_percent = 25.0
        El = 7.0
        psi_l_kgo = -2.40105971983
        J_w = El * huntingford2015_equations.MOLAR_MASS_WATER_IN_G_PER_MOL
        psi_l = huntingford2015_equations.calc_Psi_l(soil_water_percent, soil, J_w=J_w)
        np.testing.assert_allclose(psi_l, psi_l_kgo)
        
        soil_water_percent = 25.0
        El = 100.0
        psi_l_kgo = -2.40105971983
        J_w = El * huntingford2015_equations.MOLAR_MASS_WATER_IN_G_PER_MOL
        psi_l = huntingford2015_equations.calc_Psi_l(soil_water_percent, soil, J_w=J_w)
        self.assertTrue(np.isnan(psi_l))


class TestWrongEquations(unittest.TestCase):       
    def test_fill_soil_with_wrong_parameters(self):
        soil = soil_cosby_parameters.SoilType(f_sand=0.27, f_silt=0.5, f_clay=0.23)
        soil = huntingford2015_equations.fill_soil_with_wrong_parameters(soil) 
        
        # numbers from Cox et al 1999 
        # http://link.springer.com/article/10.1007%2Fs003820050276
        soil_kgo = soil_cosby_parameters.SoilType(f_sand=0.27, f_silt=0.5, f_clay=0.23)
        soil_kgo.jules_soil_parameters['b'] = 6.63 
        soil_kgo.jules_soil_parameters['sathh'] = 4.95E-2
        soil_kgo.jules_soil_parameters['satcon'] = 4.72E-3
        soil_kgo.jules_soil_parameters['sm_wilt'] = 0.136
        soil_kgo.jules_soil_parameters['sm_crit'] = 0.242
        soil_kgo.jules_soil_parameters['sm_sat'] = 0.458       
        
        for key in soil_kgo.jules_soil_parameters.keys():
            np.testing.assert_allclose(
                soil.jules_soil_parameters[key],
                soil_kgo.jules_soil_parameters[key], rtol=5e-03)
        
        

if __name__ == '__main__':
    unittest.main()
