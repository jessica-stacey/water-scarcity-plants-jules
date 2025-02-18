# -*- coding: iso-8859-1 -*-

import unittest
import numpy as np
import soil_cosby_parameters


class TestSoilCosbyParameters(unittest.TestCase):

    def test_soil_textural_class_sum(self):
        for val in soil_cosby_parameters.SOIL_TEXTURAL_CLASS.values():
            self.assertEqual(sum(val), 100)


class TestSoilType(unittest.TestCase):

    def setUp(self):

        self.soil_textural_class = 'Silty clay loam'
        self.f_sand = 0.10
        self.f_silt = 0.56
        self.f_clay = 0.34
        self.f_organic = 0.5
        self.soc_layer = 1

        # KGO for f_organic=0.0 and silty clay loam,
        # in same order as in var
        self.nonorganic_const_val = {'b': 8.408, 'hcap': 1.156E+06,
                                     'sm_wilt': 0.2488, 'hcon': 0.2055,
                                     'sm_crit': 0.3917, 'satcon': 0.00144,
                                     'sathh': 0.6278, 'sm_sat': 0.4782, 
                                     'albsoil': np.nan, 'clay':self.f_clay}

        # KGO for top layer, f_organic=0.5 and silty clay loam,
        # in same order as in var
        self.organic_const_val = {'b': 5.554, 'hcap': 8.678E+05,
                                  'sm_wilt': 0.1808, 'hcon': 0.111,
                                  'sm_crit': 0.3595, 'satcon': 0.02008,
                                  'sathh': 0.08041, 'sm_sat': 0.7041, 
                                  'albsoil': np.nan, 'clay':self.f_clay}

    def test_init_nonorganic_SoilType_with_soil_textural_class(self):
        # specify soil textural class

        soil = soil_cosby_parameters.SoilType(
            soil_textural_class=self.soil_textural_class)

        self.assertEqual(self.f_sand, soil.f_sand)
        self.assertEqual(self.f_silt, soil.f_silt)
        self.assertEqual(self.f_clay, soil.f_clay)

        for v in soil.jules_soil_parameters:
            np.testing.assert_allclose(
                soil.jules_soil_parameters[v],
                self.nonorganic_const_val[v], rtol=5e-03)

    def test_init_nonorganic_SoilType_with_sand_silt_clay_fracs(self):
        # specify f_clay, f_sand, f_silt

        soil = soil_cosby_parameters.SoilType(
            f_clay=self.f_clay, f_sand=self.f_sand, f_silt=self.f_silt)

        self.assertEqual(self.f_sand, soil.f_sand)
        self.assertEqual(self.f_silt, soil.f_silt)
        self.assertEqual(self.f_clay, soil.f_clay)

        for v in soil.jules_soil_parameters:
            np.testing.assert_allclose(
                soil.jules_soil_parameters[v],
                self.nonorganic_const_val[v], rtol=5e-03)

    def test_init_nonorganic_SoilType_with_incorrect_args(self):

        # specifying both soil textural class and f_sand
        with self.assertRaises(ValueError):
            soil = soil_cosby_parameters.SoilType(
                soil_textural_class=self.soil_textural_class,
                f_sand=self.f_sand)

        # specifying neither soil textural class nor f_clay, f_sand, f_silt
        with self.assertRaises(ValueError):
            soil = soil_cosby_parameters.SoilType()

        # f_clay, f_sand, f_silt do not add up to 1
        with self.assertRaises(ValueError):
            soil = soil_cosby_parameters.SoilType(
                soil_textural_class=self.soil_textural_class,
                f_clay=self.f_clay, f_sand=self.f_sand,
                f_silt=self.f_silt - 0.1)

    def test_init_SoilType_with_zero_organic_frac(self):
        # specify zero organic fraction

        soil = soil_cosby_parameters.SoilType(
            soil_textural_class=self.soil_textural_class,
            f_organic=0.0,
            soc_layer=1)

        for v in soil.jules_soil_parameters:
            np.testing.assert_allclose(soil.jules_soil_parameters[v],
                                       self.nonorganic_const_val[v],
                                       rtol=5e-03)

    def test_init_SoilType_with_organic_frac_of_one(self):
        # creating an entirely organic SoilType should reproduce
        # the values in soil_cosby_parameters.ORGANIC_SOIL

        sl = self.soc_layer - 1

        soil = soil_cosby_parameters.SoilType(
            soil_textural_class=self.soil_textural_class, f_organic=1.0,
            soc_layer=self.soc_layer)

        for v in ['b', 'hcap', 'hcon', 'satcon', 'sathh', 'sm_sat']:
            np.testing.assert_allclose(
                soil.jules_soil_parameters[v],
                soil_cosby_parameters.ORGANIC_SOIL[v][sl], rtol=5e-03)

        for v in ['sm_wilt', 'sm_crit']:
            np.testing.assert_allclose(
                soil.jules_soil_parameters[v],
                soil_cosby_parameters.ORGANIC_SOIL[v][sl], atol=5e-03)

    def test_init_SoilType_with_organic_frac(self):

        soil = soil_cosby_parameters.SoilType(
            soil_textural_class=self.soil_textural_class,
            f_organic=self.f_organic,
            soc_layer=self.soc_layer)

        self.assertEqual(self.f_organic, soil.f_organic)
        self.assertEqual(self.soc_layer, soil.soc_layer)

        for v in soil.jules_soil_parameters:
            np.testing.assert_allclose(
                soil.jules_soil_parameters[v],
                self.organic_const_val[v], rtol=5e-03)

    def test_init_SoilType_with_incorrect_f_organic(self):

        with self.assertRaises(ValueError):
            soil = soil_cosby_parameters.SoilType(
                soil_textural_class=self.soil_textural_class,
                f_organic=-1.0,
                soc_layer=self.soc_layer)

        with self.assertRaises(ValueError):
            soil = soil_cosby_parameters.SoilType(
                soil_textural_class=self.soil_textural_class,
                f_organic=2.0,
                soc_layer=self.soc_layer)

    def test_init_SoilType_with_incorrect_soc_layer(self):

        with self.assertRaises(ValueError):
            soil = soil_cosby_parameters.SoilType(
                soil_textural_class=self.soil_textural_class,
                f_organic=self.f_organic,
                soc_layer=0)

        with self.assertRaises(ValueError):
            soil = soil_cosby_parameters.SoilType(
                soil_textural_class=self.soil_textural_class,
                f_organic=self.f_organic,
                soc_layer=4)
                
    def test_init_SoilType_with_cap_cutoff(self):
    
        # First check values are unchanged when 
        # l_cap_cutoffs=True but soil is not organic or very fine
        
        soil = soil_cosby_parameters.SoilType(
            soil_textural_class=self.soil_textural_class,
            l_cap_cutoffs=True)

        for v in soil.jules_soil_parameters:
            np.testing.assert_allclose(
                soil.jules_soil_parameters[v],
                self.nonorganic_const_val[v], rtol=5e-03)
                
        # Now check cutoff is implemented properly for an
        # organic and very-fine soil
        
        kgo_hcon = 0.21828344443728268
        kgo_hcap = 1228468.98   
                
        organic_soil = soil_cosby_parameters.SoilType(
            soil_textural_class=self.soil_textural_class,
            f_organic=self.f_organic,
            l_cap_cutoffs=True)
            
        very_fine_soil = soil_cosby_parameters.SoilType(
            f_sand=0.05, f_clay=0.8, f_silt=0.15,
            l_cap_cutoffs=True)

        for soil in [organic_soil, very_fine_soil]:
        
            for v in soil.jules_soil_parameters:
                if v in soil_cosby_parameters.PARAMETERS_HWSD_VG_VERY_FINE:
                    np.testing.assert_allclose(
                        soil.jules_soil_parameters[v],
                        soil_cosby_parameters.PARAMETERS_HWSD_VG_VERY_FINE[v])
                elif v == 'hcon':  
                    np.testing.assert_allclose(
                        soil.jules_soil_parameters[v], kgo_hcon)
                elif v == 'hcap':    
                    np.testing.assert_allclose(
                        soil.jules_soil_parameters[v], kgo_hcap)
                elif v in ['albedo']:
                    np.testing.assert_allclose(
                        soil.jules_soil_parameters[v],
                        self.organic_const_val[v], rtol=5e-03)
                
        with self.assertRaises(ValueError):
            soil = soil_cosby_parameters.SoilType(
                soil_textural_class=self.soil_textural_class,
                f_organic=self.f_organic,
                l_cap_cutoffs=True,
                soc_layer=1)

    def test_brooks_and_corey_equation(self):
        abs_psi = 1.5E6
        target_theta = self.nonorganic_const_val['sm_wilt']
        theta_sat = self.nonorganic_const_val['sm_sat']
        theta_res = 0.0
        b = self.nonorganic_const_val['b']
        sathh = self.nonorganic_const_val['sathh']

        theta = soil_cosby_parameters.brooks_and_corey_equation(
             abs_psi=abs_psi, theta_sat=theta_sat, 
             theta_res=theta_res, b=b, sathh=sathh)

        np.testing.assert_allclose(theta,
            target_theta,
            rtol=5e-03)

        # checking it works with numpy arrays too
        shape = (2,3)
        theta_sat_arr = np.full(shape, theta_sat)
        theta_res_arr = np.full(shape, theta_res)
        b_arr = np.full(shape, b)
        sathh_arr = np.full(shape, sathh)

        shape_big = (11,2,3)
        target_theta_arr = np.full(shape_big, target_theta)
        abs_psi_arr = np.full(shape_big, abs_psi)
        theta_arr = soil_cosby_parameters.brooks_and_corey_equation(
             abs_psi=abs_psi_arr, theta_sat=theta_sat_arr, 
             theta_res=theta_res_arr, b=b_arr, sathh=sathh_arr)

        np.testing.assert_allclose(theta_arr,
            target_theta_arr,
            rtol=5e-03)

        # example where the arrays can not be broadcast together
        shape_big = (2,3,11)
        target_theta_arr = np.full(shape_big, target_theta)
        abs_psi_arr = np.full(shape_big, abs_psi)

        with self.assertRaises(ValueError):
            theta_arr = soil_cosby_parameters.brooks_and_corey_equation(
                abs_psi=abs_psi_arr, theta_sat=theta_sat_arr, 
                theta_res=theta_res_arr, b=b_arr, sathh=sathh_arr)


    def test_inverse_brooks_and_corey_equation(self):
        target_abs_psi = 1.5E6
        theta = self.nonorganic_const_val['sm_wilt']
        theta_sat = self.nonorganic_const_val['sm_sat']
        theta_res = 0.0
        b = self.nonorganic_const_val['b']
        sathh = self.nonorganic_const_val['sathh']

        abs_psi = soil_cosby_parameters.inverse_brooks_and_corey_equation(
             theta, theta_sat=theta_sat, 
             theta_res=theta_res, b=b, sathh=sathh)

        np.testing.assert_allclose(abs_psi,
            target_abs_psi,
            rtol=5e-03)

        # checking it works with numpy arrays too
        shape = (2,3)
        theta_sat_arr = np.full(shape, theta_sat)
        theta_res_arr = np.full(shape, theta_res)
        b_arr = np.full(shape, b)
        sathh_arr = np.full(shape, sathh)

        shape_big = (11,2,3)
        target_abs_psi_arr = np.full(shape_big, target_abs_psi)
        theta_arr = np.full(shape_big, theta)
        abs_psi_arr = soil_cosby_parameters.inverse_brooks_and_corey_equation(
             theta_arr, theta_sat=theta_sat_arr, 
             theta_res=theta_res_arr, b=b_arr, sathh=sathh_arr)

        np.testing.assert_allclose(abs_psi_arr,
            target_abs_psi_arr,
            rtol=5e-03)

        # example where the arrays can not be broadcast together
        shape_big = (2,3,11)
        target_abs_psi_arr = np.full(shape_big, target_abs_psi)
        theta_arr = np.full(shape_big, theta)

        with self.assertRaises(ValueError):
            abs_psi_arr = soil_cosby_parameters.inverse_brooks_and_corey_equation(
                theta_arr, theta_sat=theta_sat_arr, 
                theta_res=theta_res_arr, b=b_arr, sathh=sathh_arr)
                

class TestCalcBAndSathh(unittest.TestCase):

    def test_all_soil_textural_classes(self):
        for key in soil_cosby_parameters.SOIL_TEXTURAL_CLASS.keys():
            soil = soil_cosby_parameters.SoilType(soil_textural_class=key)
            
            b = soil_cosby_parameters.calc_b_from_sm_wilt_sm_crit(
                sm_wilt=soil.jules_soil_parameters['sm_wilt'],
                sm_crit=soil.jules_soil_parameters['sm_crit'])
                
            np.testing.assert_allclose(b, soil.jules_soil_parameters['b'])
            
            sathh = soil_cosby_parameters.sathh_from_sm_wilt_sm_crit_sm_sat(
                sm_wilt=soil.jules_soil_parameters['sm_wilt'],
                sm_sat=soil.jules_soil_parameters['sm_sat'], 
                sm_crit=soil.jules_soil_parameters['sm_crit'])
                
            np.testing.assert_allclose(sathh, soil.jules_soil_parameters['sathh'])
            
            f_clay, f_sand, f_silt = soil_cosby_parameters.soil_texture_from_sm_wilt_sm_crit(
                sm_wilt=soil.jules_soil_parameters['sm_wilt'],
                sm_crit=soil.jules_soil_parameters['sm_crit'])
                  
            np.testing.assert_allclose(f_clay, soil.f_clay)
            np.testing.assert_allclose(f_sand, soil.f_sand)
            np.testing.assert_allclose(f_silt, soil.f_silt)


if __name__ == '__main__':
    unittest.main()
