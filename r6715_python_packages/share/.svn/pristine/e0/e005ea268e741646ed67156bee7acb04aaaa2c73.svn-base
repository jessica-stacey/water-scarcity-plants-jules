# -*- coding: iso-8859-1 -*-
'''
Unittests for functions in soil_water_stress_factors.py

Note that this is work-in-progress and most of the 
functions have not have tests written for them yet.

Karina Williams
'''
import unittest
import math

import numpy as np

from soil_water_stress_factors import *


class TestJulesSoilFactorFsmcShape0(unittest.TestCase):
    def test_simple_examples(self): 
    
        soil = soil_cosby_parameters.SoilType(soil_textural_class='Loam')
        soil.jules_soil_parameters['sm_sat'] = 0.5
        soil.jules_soil_parameters['sm_crit'] = 0.4
        soil.jules_soil_parameters['sm_wilt'] = 0.2
        
        soil_water_percent_arr = np.array([0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]) * 100.0
        
        p0 = 0.0        
        fsmc_kgo_arr = [0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0]
        
        for i in range(len(soil_water_percent_arr)):
            fsmc = jules_soil_factor_fsmc_shape_0(soil_water_percent_arr[i], soil, p0=p0)
            self.assertTrue(np.allclose(fsmc, fsmc_kgo_arr[i]))
            
        p0 = 0.25        
        fsmc_kgo_arr = [0.0, 0.0, 1.0/3.0, 2.0/3.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        
        for i in range(len(soil_water_percent_arr)):
            fsmc = jules_soil_factor_fsmc_shape_0(soil_water_percent_arr[i], soil, p0=p0)
            self.assertTrue(np.allclose(fsmc, fsmc_kgo_arr[i]))


class TestCalcRootFracExponential(unittest.TestCase):
    def test_simple_example(self):
        rootd = 0.2
        dz = np.array([0.2, 0.3])
        f_root = calc_root_frac_exponential(rootd=rootd, dz=dz)
        
        f_root_kgo = np.array([(math.exp(-0.0/rootd) - math.exp(-0.2/rootd)), 
                               (math.exp(-0.2/rootd) - math.exp(-0.5/rootd))
                              ]) / (1.0 - math.exp(-np.sum(dz)/rootd))
                              
        self.assertTrue(np.allclose(f_root, f_root_kgo)) 
        
    def test_realistic_examples(self): 
        # KGO from JULES 4.9
        dz = np.array([0.1, 0.25, 0.65, 2.0])
        rootd = 0.5
        f_root_kgo = np.array([0.1817197, 0.3229460, 0.3621477, 0.1331867]) 
        f_root = calc_root_frac_exponential(rootd=rootd, dz=dz)
        self.assertTrue(np.allclose(f_root, f_root_kgo)) 
        
        rootd = 1.0
        f_root_kgo = np.array([0.1001487, 0.2106363, 0.3544559, 0.3347590])
        f_root = calc_root_frac_exponential(rootd=rootd, dz=dz)
        self.assertTrue(np.allclose(f_root, f_root_kgo)) 
        
        rootd = 3.0
        f_root_kgo = np.array([5.1863406E-02, 0.1223410, 0.2742365, 0.5515591])
        f_root = calc_root_frac_exponential(rootd=rootd, dz=dz)
        self.assertTrue(np.allclose(f_root, f_root_kgo)) 


if __name__ == '__main__':
    unittest.main()