# -*- coding: iso-8859-1 -*-
import unittest
import os
import datetime

import numpy as np

import netcdftime
import iris
import iris.tests
import iris.tests.stock

from make_time_coord import wrap_unit
import fluxnet_evaluation

if iris.__version__ == '1.8.1':
    from realistic_3d import realistic_3d
else:
    from iris.tests.stock import realistic_3d

'''
Contains tests for the fluxnet_evaluation script.

Karina Williams
'''


fluxnet_evaluation.OBS_FOLDER_FLUXNET2015 = 'testdata'
fluxnet_evaluation.OBS_FOLDER_PRE2015 = 'testdata/preFLUXNET2015'

MODEL_FOLDER = os.path.expandvars('testdata/model_output')


class TestSimpleExamples(unittest.TestCase):
    def test_reduce_month_coord_string(self):
        self.assertEqual(fluxnet_evaluation.reduce_month_coord_string('Jan|Jan|Jan|Jan'), 'Jan')
        self.assertEqual(fluxnet_evaluation.reduce_month_coord_string('Jan|Feb|Jan|Jan'), 'Jan|Feb|Jan|Jan')
        self.assertEqual(fluxnet_evaluation.reduce_month_coord_string('Jan'), 'Jan')
 

class TestIrisUtilsUnifyTimeUnitsBehaviour(iris.tests.IrisTest):
    def test_behaves_as_expected_same_calendar(self):
        cube_a = realistic_3d()
        cube_b = cube_a.copy()
        new_unit = wrap_unit('seconds since 1999-01-01 00:00:00', 'gregorian')
        cube_a.coord('time').convert_units(new_unit)
        new_unit = wrap_unit('seconds since 2002-01-01 00:00:00', 'gregorian')
        cube_b.coord('time').convert_units(new_unit)
        iris.util.unify_time_units([cube_a, cube_b])
        self.assertEqual(str(cube_b.coord('time').units), 'seconds since 1999-01-01 00:00:00')
        
    def test_behaves_as_expected_different_calendar(self):
        cube_a = realistic_3d()
        new_unit = wrap_unit('seconds since 1999-01-01 00:00:00', 'standard')
        cube_a.coord('time').units = new_unit
        cube_b = cube_a.copy()
        new_unit = wrap_unit('seconds since 2002-01-01 00:00:00', 'gregorian')
        cube_b.coord('time').units = new_unit
        iris.util.unify_time_units([cube_a, cube_b])
        self.assertNotEqual(str(cube_b.coord('time').units), 'seconds since 1999-01-01 00:00:00') # but this should be!
               

class TestUnifyTimeUnits(iris.tests.IrisTest):
    def test_behaves_as_expected_same_calendar(self):
        cube_a = realistic_3d()
        cube_b = cube_a.copy()
        new_unit = wrap_unit('seconds since 1999-01-01 00:00:00', 'gregorian')
        cube_a.coord('time').convert_units(new_unit)
        new_unit = wrap_unit('seconds since 2002-01-01 00:00:00', 'gregorian')
        cube_b.coord('time').convert_units(new_unit)
        fluxnet_evaluation.unify_time_units([cube_a, cube_b])
        self.assertEqual(str(cube_b.coord('time').units), 'seconds since 1999-01-01 00:00:00')
        
    def test_behaves_as_expected_different_calendar(self):
        cube_a = realistic_3d()
        new_unit = wrap_unit('seconds since 1999-01-01 00:00:00', 'standard')
        cube_a.coord('time').units = new_unit
        cube_b = cube_a.copy()
        new_unit = wrap_unit('seconds since 2002-01-01 00:00:00', 'gregorian')
        cube_b.coord('time').units = new_unit
        self.assertRaises(UserWarning, fluxnet_evaluation.unify_time_units, [cube_a, cube_b]) 
        

class TestWithRealSubdailyGPPFiles(unittest.TestCase):
    def setUp(self):
        simulation = "vn4.6_trunk_for_testing"
        site = "JP_Tak"
        var = 'GPP'
        run_id = site + '-' + simulation
        self.subdaily_obs_cube = fluxnet_evaluation.read_subdaily_pre2015_obs(site, 
            fluxnet_evaluation.VAR_NAMES_DICT[var]['obs_var_name_pre2015'])
        self.subdaily_model_cube = fluxnet_evaluation.read_model_output(
            fluxnet_evaluation.VAR_NAMES_DICT[var]['model_var_name'],  
            MODEL_FOLDER, run_id=run_id, profile_name='H')

    def tearDown(self):
        self.subdaily_obs_cube = None
        self.subdaily_model_cube = None

    def test_time_units_are_equal(self):
        # time units do not start off equal for the JP_Tak site
        self.assertFalse(fluxnet_evaluation.time_units_are_equal([self.subdaily_obs_cube, self.subdaily_model_cube]))
        fluxnet_evaluation.unify_time_units([self.subdaily_obs_cube, self.subdaily_model_cube])
        self.assertTrue(fluxnet_evaluation.time_units_are_equal([self.subdaily_obs_cube, self.subdaily_model_cube]))
    
    def test_chop_off_partial_days(self):
        # take a slice of the subdaily obs cube, with a partial day at either end
        new_cube = self.subdaily_obs_cube[5:100]

        with iris.FUTURE.context(cell_datetime_objects=True):
            # check the cube times are as expected before any chopping:
            self.assertEqual(new_cube.coord('time').cell(0).bound[0], netcdftime.datetime(1999, 1, 1, 2, 30, 0))
            self.assertEqual(new_cube.coord('time').cell(-1).bound[1], netcdftime.datetime(1999, 1, 3, 2, 0, 0))

            new_cube = fluxnet_evaluation.chop_off_partial_days(new_cube)

            # check the cube times are as expected after chopping:
            self.assertEqual(new_cube.coord('time').cell(0).bound[0], netcdftime.datetime(1999, 1, 2, 0, 0, 0))
            self.assertEqual(new_cube.coord('time').cell(-1).bound[1], netcdftime.datetime(1999, 1, 3, 0, 0, 0))

        new_cube = self.subdaily_obs_cube[5:200]
        new_cube.data = np.ma.MaskedArray(new_cube.data) # because it wasn't a masked array at the beginning
        new_cube.data[0:48] = np.ma.masked 

        with iris.FUTURE.context(cell_datetime_objects=True):
            # check the cube times are as expected before any chopping:
            self.assertEqual(new_cube.coord('time').cell(0).bound[0], netcdftime.datetime(1999, 1, 1, 2, 30, 0))
            self.assertEqual(new_cube.coord('time').cell(-1).bound[1], netcdftime.datetime(1999, 1, 5, 4, 0, 0))

            new_cube_2 = fluxnet_evaluation.chop_off_partial_days(new_cube)

            # check the cube times are as expected after chopping:
            self.assertEqual(new_cube_2.coord('time').cell(0).bound[0], netcdftime.datetime(1999, 1, 3, 0, 0, 0))
            self.assertEqual(new_cube_2.coord('time').cell(-1).bound[1], netcdftime.datetime(1999, 1, 5, 0, 0, 0))
            
            new_cube_3 = fluxnet_evaluation.chop_off_partial_days(new_cube, check_masked=False)

            # check the cube times are as expected after chopping:
            self.assertEqual(new_cube_3.coord('time').cell(0).bound[0], netcdftime.datetime(1999, 1, 2, 0, 0, 0))
            self.assertEqual(new_cube_3.coord('time').cell(-1).bound[1], netcdftime.datetime(1999, 1, 5, 0, 0, 0))


    def test_chop_off_partial_months(self):
        # take a slice of the subdaily obs cube, with a partial day at either end
        new_cube = self.subdaily_model_cube[5:2000]
        
        with iris.FUTURE.context(cell_datetime_objects=True):
            # check the cube times are as expected before any chopping:
            self.assertEqual(new_cube.coord('time').cell(0).bound[0], netcdftime.datetime(2002, 1, 1, 5, 0, 0))
            self.assertEqual(new_cube.coord('time').cell(-1).bound[1], netcdftime.datetime(2002, 3, 25, 8, 0, 0))

            new_cube = fluxnet_evaluation.chop_off_partial_months(new_cube)

            # check the cube times are as expected after chopping:
            self.assertEqual(new_cube.coord('time').cell(0).bound[0], netcdftime.datetime(2002, 2, 1, 0, 0, 0))
            self.assertEqual(new_cube.coord('time').cell(-1).bound[1], netcdftime.datetime(2002, 3, 1, 0, 0, 0))

        new_cube = self.subdaily_model_cube[5:2800]
        new_cube.data = np.ma.MaskedArray(new_cube.data) # because it wasn't a masked array at the beginning
        new_cube.data[0:740] = np.ma.masked 
        
        with iris.FUTURE.context(cell_datetime_objects=True):
            # check the cube times are as expected before any chopping:
            self.assertEqual(new_cube.coord('time').cell(0).bound[0], netcdftime.datetime(2002, 1, 1, 5, 0, 0))
            self.assertEqual(new_cube.coord('time').cell(-1).bound[1], netcdftime.datetime(2002, 4, 27, 16, 0))

            new_cube = fluxnet_evaluation.chop_off_partial_months(new_cube)

            # check the cube times are as expected after chopping:
            self.assertEqual(new_cube.coord('time').cell(0).bound[0], netcdftime.datetime(2002, 3, 1, 0, 0, 0))
            self.assertEqual(new_cube.coord('time').cell(-1).bound[1], netcdftime.datetime(2002, 4, 1, 0, 0, 0))
        
    def test_check_all_time_coords_are_the_same(self):
        new_cube1 = self.subdaily_obs_cube[5:100]

        new_cube2 = new_cube1.copy()
        self.assertTrue(fluxnet_evaluation.check_all_time_coords_are_the_same([new_cube1, new_cube2]))

        new_cube2 = new_cube2[1:]
        self.assertFalse(fluxnet_evaluation.check_all_time_coords_are_the_same([new_cube1, new_cube2]))

        new_cube2 = new_cube1.copy()
        new_cube2.coord('time').bounds = None
        self.assertFalse(fluxnet_evaluation.check_all_time_coords_are_the_same([new_cube1, new_cube2]))

        new_cube2 = new_cube1.copy()
        new_cube2.coord('time').bounds = None
        new_cube2.coord('time').guess_bounds(bound_position=0.0)
        self.assertTrue(fluxnet_evaluation.check_all_time_coords_are_the_same([new_cube1, new_cube2]))

        new_cube2 = new_cube1.copy()
        new_cube2.coord('time').bounds = None
        new_cube2.coord('time').guess_bounds(bound_position=0.5)
        self.assertFalse(fluxnet_evaluation.check_all_time_coords_are_the_same([new_cube1, new_cube2]))

        new_cube2 = new_cube1.copy()
        new_cube2.coord('time').units = None
        self.assertFalse(fluxnet_evaluation.check_all_time_coords_are_the_same([new_cube1, new_cube2]))

    def test_cut_to_overlapping_time_period(self):
        new_cube1 = self.subdaily_obs_cube[5:100]
        new_cube2 = self.subdaily_obs_cube[2:80]
        new_cube3 = self.subdaily_obs_cube[5:70]

        cubes = [new_cube1, new_cube2, new_cube3]

        cubes = fluxnet_evaluation.cut_to_overlapping_time_period(cubes)

        self.assertEqual(cubes[0], new_cube3)
        self.assertEqual(cubes[1], new_cube3)
        self.assertEqual(cubes[2], new_cube3)

        # non-overlapping example
        new_cube1 = self.subdaily_obs_cube[2:6]
        new_cube2 = self.subdaily_obs_cube[10:16]

        cubes = [new_cube1, new_cube2]

        cubes = fluxnet_evaluation.cut_to_overlapping_time_period(cubes)

        self.assertEqual(cubes, iris.cube.CubeList([]))

    def test_masking_on_obs_files(self):
        self.assertEqual(np.min(self.subdaily_obs_cube.data), 0.0)
        self.assertTrue(isinstance(self.subdaily_obs_cube.data, np.ma.MaskedArray))
        self.assertEqual(np.ma.count_masked(self.subdaily_obs_cube.data), 10362)

class TestWithSubdailyLEFilesPre2015(unittest.TestCase):
    def setUp(self):
        site = "JP_Tak"
        var = 'LE'
        self.subdaily_obs_cube = fluxnet_evaluation.read_subdaily_pre2015_obs(site, 
            fluxnet_evaluation.VAR_NAMES_DICT[var]['obs_var_name_pre2015'])

    def tearDown(self):
        self.subdaily_obs_cube = None

    def test_masking_on_obs_files(self):
        cube = self.subdaily_obs_cube.copy()
        self.assertEqual(np.min(cube.data), -277.0)
        self.assertTrue(isinstance(cube.data, np.ma.MaskedArray))
        self.assertEqual(np.ma.count_masked(cube.data), 25955)
        self.assertEqual(np.ma.count(cube.data), 79261)

        iris.coord_categorisation.add_year(cube, 'time')
        iris.coord_categorisation.add_day_of_year(cube, 'time')

        cube = cube.aggregated_by(["year", "day_of_year"], iris.analysis.MEAN)

        self.assertTrue(np.min(cube.data) > -16.0)


class TestWithSubdailyLEFilesFluxnet2015(unittest.TestCase):
    def test_against_kgo(self):
        site = "FI_Hyy_small_for_testing"
        var = 'LE'
        cube = fluxnet_evaluation. get_subdaily_fluxnet2015_obs_data(site, var)
        print cube
        # check data is as expected
        self.assertEqual(np.max(cube.data), 69.305)
        self.assertEqual(np.min(cube.data), -2.4975)
        self.assertEqual(np.ma.count_masked(cube.data), 0)

        # check first element of time coord is in format expected 
        # (if not in ths format, code will complain when it compares it to the model cube)
        time_units = cube.coord('time').units
        self.assertEqual(time_units, 
            wrap_unit("hours since 1970-01-01 00:00:00", calendar='gregorian'))
        point = time_units.date2num(datetime.datetime(1996, 1, 1))
        bounds = np.array([[point, point + 0.5]])
        kgo_time_coord_0 = iris.coords.DimCoord(
            [point],
            long_name='Time of data',
            var_name='time',
            standard_name='time',
            units=time_units
            )

        # need to test bounds separately as they're not quite equal
        np.testing.assert_array_almost_equal(cube.coord('time')[0].bounds, bounds)
        cube.coord('time').bounds = None
        self.assertEqual(cube.coord('time')[0], kgo_time_coord_0)

        # check last point is as expected
        self.assertEqual(cube.coord('time').points[-1], 
            time_units.date2num(datetime.datetime(1996, 1, 5, 8, 30)))


class TestBowenRatioFromFluxnet2015(unittest.TestCase):
    def test_against_kgo(self):
        site = "FI_Hyy_small_for_testing"
        var = 'Bowen_Ratio'
        cube_a = fluxnet_evaluation.get_daily_fluxnet2015_obs_data(site, var, regenerate_files=True, local_time=True)
        cube_b = fluxnet_evaluation.get_daily_fluxnet2015_obs_data(site, var, regenerate_files=False, local_time=True)
        
        for cube in [cube_a, cube_b]:
            # check data is as expected
            self.assertTrue(np.max(cube.data), 2.033606296260857)
            self.assertTrue(np.min(cube.data), -1.0405955666296196)
            self.assertEqual(np.ma.count_masked(cube.data), 0)

            # check first element of time coord is in format expected 
            # (if not in ths format, code will complain when it compares it to the model cube)
            time_units = cube.coord('time').units
            self.assertEqual(time_units, 
                wrap_unit("hours since 1970-01-01 00:00:00", calendar='gregorian'))
            point = time_units.date2num(datetime.datetime(1996, 1, 1))
            kgo_time_coord_0 = iris.coords.DimCoord(
                [point + 12],
                bounds=[[point, point + 24]],
                long_name='Time of data',
                var_name='time',
                standard_name='time',
                units=time_units
                )
            self.assertEqual(cube.coord('time')[0], kgo_time_coord_0)

            # check last point is as expected (1996-01-05 gets chopped off as not complete day)
            self.assertEqual(cube.coord('time').points[-1], 
                time_units.date2num(datetime.datetime(1996, 1, 4, 12)))

            self.assertEqual(cube.units, wrap_unit('1'))
            
            for coord_str in ['year', 'day_of_year', 'month']:
                self.assertTrue(cube.coord(coord_str).var_name is None)


class TestWithDailyLEFilesFluxnet2015(unittest.TestCase):
    def test_against_kgo(self):
        site = "FI_Hyy"
        var = 'LE'
        local_time = False
        cube = fluxnet_evaluation.get_daily_fluxnet2015_obs_data(site, var, local_time=local_time)

        # check data is as expected
        self.assertEqual(np.max(cube.data), 125.654721458)
        self.assertEqual(np.min(cube.data), -18.0614861667)
        self.assertEqual(np.ma.count_masked(cube.data), 0)

        # check first element of time coord is in format expected 
        # (if not in ths format, code will complain when it compares it to the model cube)
        time_units = cube.coord('time').units
        self.assertEqual(time_units, 
            wrap_unit("hours since 1970-01-01 00:00:00", calendar='gregorian'))
        point = time_units.date2num(datetime.datetime(1996, 1, 1))
        kgo_time_coord_0 = iris.coords.DimCoord(
            [point + 12],
            bounds=[[point, point + 24]],
            long_name='Time of data',
            var_name='time',
            standard_name='time',
            units=time_units
            )
        self.assertEqual(cube.coord('time')[0], kgo_time_coord_0)

        # check last point is as expected
        self.assertEqual(cube.coord('time').points[-1], 
            time_units.date2num(datetime.datetime(2014, 12, 30, 12)))

        self.assertEqual(cube.units, wrap_unit('W m-2'))
        
        # Now again with local_time=True
        local_time = True
        cube = fluxnet_evaluation.get_daily_fluxnet2015_obs_data(site, var, local_time=local_time)

        # check data is as expected
        self.assertEqual(np.max(cube.data), 125.326)
        self.assertEqual(np.min(cube.data), -16.9925)
        self.assertEqual(np.ma.count_masked(cube.data), 0)

        # check first element of time coord is in format expected 
        # (if not in ths format, code will complain when it compares it to the model cube)
        time_units = cube.coord('time').units
        self.assertEqual(time_units, 
            wrap_unit("hours since 1970-01-01 00:00:00", calendar='gregorian'))
        point = time_units.date2num(datetime.datetime(1996, 1, 1))
        kgo_time_coord_0 = iris.coords.DimCoord(
            [point + 12],
            bounds=[[point, point + 24]],
            long_name='Time of data',
            var_name='time',
            standard_name='time',
            units=time_units
            )
        self.assertEqual(cube.coord('time')[0], kgo_time_coord_0)

        # check last point is as expected
        self.assertEqual(cube.coord('time').points[-1], 
            time_units.date2num(datetime.datetime(2014, 12, 31, 12)))

        self.assertEqual(cube.units, wrap_unit('W m-2'))

class TestWithDailyGPPFilesPre2015(unittest.TestCase):
    def setUp(self):
        simulation = "vn4.6_trunk_for_testing"
        site = "JP_Tak"
        var = 'GPP'
        run_id = site + '-' + simulation
        self.daily_obs_cube_regenerated = fluxnet_evaluation.get_daily_pre2015_obs_data(site, 
            fluxnet_evaluation.VAR_NAMES_DICT[var]['obs_var_name_pre2015'], regenerate_files=True)
        self.daily_obs_cube_from_saved = fluxnet_evaluation.get_daily_pre2015_obs_data(site, 
            fluxnet_evaluation.VAR_NAMES_DICT[var]['obs_var_name_pre2015'], regenerate_files=False)
        self.daily_model_cube = fluxnet_evaluation.get_processed_daily_model_data( 
            fluxnet_evaluation.VAR_NAMES_DICT[var]['model_var_name'], 
            MODEL_FOLDER, run_id=run_id)

    def tearDown(self):
        self.daily_obs_cube_regenerated = None
        self.daily_obs_cube_from_saved = None
        self.daily_model_cube = None

    def test_calc_rmse(self):
        for cube1 in [self.daily_obs_cube_regenerated.copy(), self.daily_obs_cube_from_saved.copy()]:
            cube2 = self.daily_model_cube.copy()

            fluxnet_evaluation.unify_time_units([cube1, cube2]) 
            cube1, cube2 = fluxnet_evaluation.cut_to_overlapping_time_period([cube1, cube2])

            fluxnet_evaluation.check_all_time_coords_are_the_same([cube1, cube2])

            res_from_function = fluxnet_evaluation.calc_rmse(cube1, cube2)

            res_from_arrays = np.sqrt(((cube1.data - cube2.data) ** 2.0).mean())

            self.assertEqual(res_from_function, res_from_arrays)
            self.assertAlmostEqual(res_from_function, 2.5417785378763615)
            
            
class TestWithMoreRealDailyGPPFilesPre2015(unittest.TestCase):
    def setUp(self):
        simulation = "vn4.6_trunk_for_testing"
        site = "JP_Tak"
        var = 'GPP'
        run_id = site + '-' + simulation
        self.daily_obs_cube_from_saved = fluxnet_evaluation.get_daily_pre2015_obs_data(site, 
            fluxnet_evaluation.VAR_NAMES_DICT[var]['obs_var_name_pre2015'], regenerate_files=False)
        self.daily_model_cube = fluxnet_evaluation.get_processed_daily_model_data( 
            fluxnet_evaluation.VAR_NAMES_DICT[var]['model_var_name'], 
            MODEL_FOLDER, run_id=run_id, 
            use_daily_output_if_possible=True)
            
    def tearDown(self):
        self.daily_obs_cube_from_saved = None
        self.daily_model_cube = None

    def test_calc_rmse(self):
        for cube1 in [self.daily_obs_cube_from_saved.copy()]:
            cube2 = self.daily_model_cube.copy()

            fluxnet_evaluation.unify_time_units([cube1, cube2]) 
            
            cube1, cube2 = fluxnet_evaluation.cut_to_overlapping_time_period([cube1, cube2])

            fluxnet_evaluation.check_all_time_coords_are_the_same([cube1, cube2])
            
            res_from_function = fluxnet_evaluation.calc_rmse(cube1, cube2)

            res_from_arrays = np.sqrt(((cube1.data - cube2.data) ** 2.0).mean())

            self.assertEqual(res_from_function, res_from_arrays)
            self.assertAlmostEqual(res_from_function, 2.5452224439296405)


class TestGetSiteList(unittest.TestCase):
    def test_with_example_file(self):
        # this example file also contains some empty lines
        filename = 'testdata/list_sites_test.txt'
        site_list = fluxnet_evaluation.get_site_list(filename)
        site_list_kgo = ["AU_Fog", "BR_Ji2", "BR_Ma2"]
        self.assertEqual(site_list, site_list_kgo)       


class TestSimpleMaskedArrayFeatures(unittest.TestCase):
    '''
    Features of masked arrays I want to try out before using them
    '''
    def test_applying_mask_from_one_array_to_another(self):
        a = np.array([1,2,3,4])
        b = np.ma.masked_greater(a, 3)
        c = np.ma.masked_equal(a, 2)
        c = np.ma.masked_where(np.ma.getmask(b), c)
        x = np.ma.array(a, mask=[[0, 1, 0, 1]])
        np.testing.assert_equal(c, x)       
        

@unittest.skip("skipping... not part of the standard test suite")
class TestProblemSite(unittest.TestCase):
    '''
    This class is just for investigating things that look a bit odd. 
    '''
    def test_why_CA_Mer_GPP_is_low(self):
        simulation = "vn4.6_trunk"
        site = "CA_Mer"
        
        run_id = site + '-' + simulation

        for var in ['GPP', 'SH']:
            obs_cube = fluxnet_evaluation.get_daily_pre2015_obs_data(site, 
                fluxnet_evaluation.VAR_NAMES_DICT[var]['obs_var_name_pre2015'], regenerate_files=False)
            model_cube = fluxnet_evaluation.get_processed_daily_model_data( 
                fluxnet_evaluation.VAR_NAMES_DICT[var]['model_var_name'],  
                MODEL_FOLDER, run_id=run_id, 
                use_daily_output_if_possible=False)
            print var
            print np.mean(obs_cube.data)    
            print np.mean(model_cube.data)   
            print np.nanmean(model_cube.data) 
        
    def test_why_BR_Ji2_SH_has_problem(self):
        simulation = "vn4.6_trunk"
        site = "BR_Ji2"
        var = 'SH'
        run_id = site + '-' + simulation
        
        subdaily_model_cube = fluxnet_evaluation.read_model_output(
            fluxnet_evaluation.VAR_NAMES_DICT[var]['model_var_name'],  
            MODEL_FOLDER, run_id=run_id, profile_name='H')
        print np.max(subdaily_model_cube.data) #-9501.41
        print np.mean(subdaily_model_cube.data) #-9809.58
        
        site = "BR_Ma2"
        subdaily_model_cube = fluxnet_evaluation.read_model_output( 
            fluxnet_evaluation.VAR_NAMES_DICT[var]['model_var_name'],  
            MODEL_FOLDER, run_id=run_id, profile_name='H')
        print np.max(subdaily_model_cube.data) #455.908
        print np.mean(subdaily_model_cube.data) #0.254714
        

if __name__ == '__main__':

    unittest.main()

    #suite = unittest.TestSuite()
    #suite.addTest(TestWithDailyLEFilesFluxnet2015('test_against_kgo'))
    #suite.addTest(TestIrisUtilsUnifyTimeUnitsBehaviour('test_behaves_as_expected_different_calendar'))
    #suite.addTest(TestUnifyTimeUnits('test_behaves_as_expected_same_calendar'))
    #unittest.TextTestRunner(verbosity=2).run(suite)
