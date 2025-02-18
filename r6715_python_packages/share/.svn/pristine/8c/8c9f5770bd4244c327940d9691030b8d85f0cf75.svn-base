# -*- coding: iso-8859-1 -*-
'''
unit tests for calc_weights.py
'''

from __future__ import absolute_import, division, print_function

import unittest

import numpy as np

import iris

import calc_weights


class TestCalcWeights(unittest.TestCase):

    def test_examples1(self):

        dzsoil_io = np.array([0.1, 0.25, 0.65, 2.0])
        
        weights = calc_weights.calc_weights(dz=dzsoil_io, 
            required_depth_of_top_of_soil_level=0.0, 
            required_depth_of_base_of_soil_level=0.05)
        self.assertTrue(np.allclose(weights, np.array([0.5, 0.0, 0.0, 0.0])))
        
        weights = calc_weights.calc_weights(dz=dzsoil_io, 
            required_depth_of_top_of_soil_level=0.0, 
            required_depth_of_base_of_soil_level=0.1)
        self.assertTrue(np.allclose(weights, np.array([1.0, 0.0, 0.0, 0.0])))
        
        weights = calc_weights.calc_weights(dz=dzsoil_io, 
            required_depth_of_top_of_soil_level=0.0, 
            required_depth_of_base_of_soil_level=0.35)
        self.assertTrue(np.allclose(weights, np.array([1.0, 1.0, 0.0, 0.0])))
        
        weights = calc_weights.calc_weights(dz=dzsoil_io, 
            required_depth_of_top_of_soil_level=0.0, 
            required_depth_of_base_of_soil_level=0.2)
        self.assertTrue(np.allclose(weights, np.array([1.0, 0.4, 0.0, 0.0])))
        
        weights = calc_weights.calc_weights(dz=dzsoil_io, 
            required_depth_of_top_of_soil_level=0.15, 
            required_depth_of_base_of_soil_level=0.2)
        self.assertTrue(np.allclose(weights, np.array([0.0, 0.2, 0.0, 0.0])))
        
        weights = calc_weights.calc_weights(dz=dzsoil_io, 
            required_depth_of_top_of_soil_level=0.15, 
            required_depth_of_base_of_soil_level=0.48)
        self.assertTrue(np.allclose(weights, np.array([0.0, 0.8, 0.2, 0.0])))
        
        weights = calc_weights.calc_weights(dz=dzsoil_io, 
            required_depth_of_top_of_soil_level=0.15, 
            required_depth_of_base_of_soil_level=3.0)
        self.assertTrue(np.allclose(weights, np.array([0.0, 0.8, 1.0, 1.0])))
        
        with self.assertRaises(UserWarning):
            weights = calc_weights.calc_weights(dz=dzsoil_io, 
                required_depth_of_top_of_soil_level=3.0, 
                required_depth_of_base_of_soil_level=2.0)
                
        with self.assertRaises(UserWarning):
            weights = calc_weights.calc_weights(dz=dzsoil_io, 
                required_depth_of_top_of_soil_level=2.0, 
                required_depth_of_base_of_soil_level=3.1)
                
    def test_examples2(self):
        # test water conservation  
        dz_model = np.array([0.1, 0.25, 0.65, 2.0]) # adds up to 3m
        dz_required = np.array([1.0, 1.0, 1.0])  # adds up to 3m
        water_content = np.array([3.1, 2.7, 6.5, 1.9]) # in kg m-2
        
        required_depth_of_top_of_soil_level = np.array([0.0, 1.0, 2.0])
        required_depth_of_base_of_soil_level = np.array([1.0, 2.0, 3.0])
        
        total_water = 0.0
        for i in range(len(dz_required)):
            
            weights = calc_weights.calc_weights(dz=dz_model, 
                required_depth_of_top_of_soil_level=required_depth_of_top_of_soil_level[i], 
                required_depth_of_base_of_soil_level=required_depth_of_base_of_soil_level[i])
                
            total_water += np.sum(weights * water_content)
        
        np.testing.assert_allclose(total_water, np.sum(water_content))
        
    def test_examples3(self):

        dzsoil_io = np.array([0.0, 20.0, 10.0, 10.0, 10.0 ])    
        
        weights = calc_weights.calc_weights(dz=dzsoil_io, 
            required_depth_of_top_of_soil_level=0.0, 
            required_depth_of_base_of_soil_level=50.0)
                       
        print(weights)    
        self.assertTrue(np.allclose(weights, np.array([0.0, 1.0, 1.0, 1.0, 1.0])))
        
        with self.assertRaises(UserWarning): #dzsoil_io is not a np.ndarray
            dzsoil_io = [0.0, 20.0, 10.0, 10.0, 10.0 ] 
            weights = calc_weights.calc_weights(dz=dzsoil_io, 
                required_depth_of_top_of_soil_level=0.0, 
                required_depth_of_base_of_soil_level=50.0)
                
        with self.assertRaises(UserWarning):
            dzsoil_io = np.array([0, 20, 10, 10, 10]) # integer array 
            weights = calc_weights.calc_weights(dz=dzsoil_io, 
                required_depth_of_top_of_soil_level=0.0, 
                required_depth_of_base_of_soil_level=50.0)
                

class TestGetWaterContentOnNewLevels(unittest.TestCase):
    def test_example(self):
        dz_model = np.array([0.1, 0.25, 0.65, 2.0]) # adds up to 3m
        dz_required = np.array([1.0, 1.0, 1.0])  # adds up to 3m
        water_content_model = np.array([3.1, 2.7, 6.5, 1.9]) # in kg m-2
        
        water_content_new = calc_weights.get_water_content_on_new_levels(water_content_model, 
            dz_old=dz_model, dz_new=dz_required)
        
        np.testing.assert_allclose(water_content_new, np.array([12.3, 0.95, 0.95]))
        
        # test water conservation  
        total_water = np.sum(water_content_new)
        
        np.testing.assert_allclose(total_water, np.sum(water_content_model))


class TestDepthsFromVarName(unittest.TestCase):

    def test_all(self):
        var_name = 'vol_soil_water_0-30cm'
        depths = calc_weights.depths_from_var_name(var_name)
        self.assertEqual(depths, [0.0, 0.3])
        
        var_name = 'vol_soil_water_30-60cm'
        depths = calc_weights.depths_from_var_name(var_name)
        self.assertEqual(depths, [0.3, 0.6])
        
        var_name = 'vol_soil_water_60-90cm'
        depths = calc_weights.depths_from_var_name(var_name)
        self.assertEqual(depths, [0.6, 0.9])
        
        var_name = 'vol_soil_water_90-120cm'
        depths = calc_weights.depths_from_var_name(var_name)
        self.assertEqual(depths, [0.9, 1.2])
        
        var_name = 'vol_soil_water_120-150cm'
        depths = calc_weights.depths_from_var_name(var_name)
        self.assertEqual(depths, [1.2, 1.5])
        
        var_name = 'vol_soil_water_150-180cm'
        depths = calc_weights.depths_from_var_name(var_name)
        self.assertEqual(depths, [1.5, 1.8])

        var_name = 'vol_soil_water_at_30cm'
        depths = calc_weights.depths_from_var_name(var_name)
        self.assertEqual(depths, 0.3)

        var_name = 'soil_temperature_at_200cm'
        depths = calc_weights.depths_from_var_name(var_name)
        self.assertEqual(depths, 2.0)
    
        
class TestLinInterpLayers(unittest.TestCase):
    def test_simple_examples(self):

        dz = np.array([0.1, 0.25, 0.65, 2.0]) # adds up to 3m
        # mid points are [ 0.05   0.225  0.675  2.   ]
    
        def interp(x, x0, x1, y0, y1):
            return y0 + (x - x0) * (y1 - y0) / (x1 - x0)

        arr_1d = np.array([1.5, 2.0, 0.5, 0.7])

        z = 0.05
        val = calc_weights.lin_interp_layers(arr_1d, dz=dz, pick_z=z)
        self.assertAlmostEqual(val, 1.5)

        z = 2.0
        val = calc_weights.lin_interp_layers(arr_1d, dz=dz, pick_z=z)
        self.assertAlmostEqual(val, 0.7)

        z = 0.1
        val = calc_weights.lin_interp_layers(arr_1d, dz=dz, pick_z=z)
        self.assertAlmostEqual(val, interp(z, 0.05, 0.225, 1.5, 2.0))

        z = 1.5
        val = calc_weights.lin_interp_layers(arr_1d, dz=dz, pick_z=z)
        self.assertAlmostEqual(val, interp(z, 0.675, 2.0, 0.5, 0.7))

        with self.assertRaises(UserWarning):
          z = 0.04
          val = calc_weights.lin_interp_layers(arr_1d, dz=dz, pick_z=z)

        with self.assertRaises(UserWarning):
          z = 2.1
          val = calc_weights.lin_interp_layers(arr_1d, dz=dz, pick_z=z)

        # now including a 'surface' value
        z = 0.04
        val = calc_weights.lin_interp_layers(np.insert(arr_1d, 0, 5.0), dz=np.insert(dz, 0, 0.0), pick_z=z)
        self.assertAlmostEqual(val, interp(z, 0.0, 0.05, 5.0, 1.5))

        z = 0.04
        val = calc_weights.lin_interp_layers(arr_1d, dz=dz, pick_z=z, surface_val=5.0)
        self.assertAlmostEqual(val, interp(z, 0.0, 0.05, 5.0, 1.5))
       

if __name__ == '__main__':
    unittest.main()
