# -*- coding: iso-8859-1 -*-

from __future__ import absolute_import, division, print_function

import re

import numpy as np

import iris

'''

Function for calculating weights to get water content of a specified 
soil layer and other utils for dealing with soil layers.

Used to generate the prescribed soil moisture for sm-stress JPEG ssuite u-al752
and to post-process AgMIP-maize ET study runs.

-- Karina Williams 

'''


def calc_weights(dz=None, required_depth_of_top_of_soil_level=None, required_depth_of_base_of_soil_level=None):
    # these weights should be applied to something in the units of kg m-2 not kg m-3
    
    if not isinstance(dz, np.ndarray):
        raise UserWarning('dz should be a numpy ndarray')
        
    if dz.dtype == np.int:
        raise UserWarning('dz should not be an integer array')
    
    if required_depth_of_top_of_soil_level > required_depth_of_base_of_soil_level:
        raise UserWarning('have got these the wrong way round')
        
    soil_column_depth = np.sum(dz)
    
    if required_depth_of_base_of_soil_level > soil_column_depth:
        raise UserWarning('lower limit is outside the soil column')
        
    nshyd = len(dz)
    weights = np.zeros(nshyd)
    
    depth_of_base_of_soil_level = np.cumsum(dz)
    depth_of_top_of_soil_level = depth_of_base_of_soil_level - dz
    
    for i,z in enumerate(dz):
        if dz[i] <= 0.0:
            weights[i] = 0
        elif depth_of_base_of_soil_level[i] < required_depth_of_top_of_soil_level: 
            # no overlap and required level is below model level
            weights[i] = 0
        elif depth_of_top_of_soil_level[i] > required_depth_of_base_of_soil_level:  
            # no overlap and required level is above model level
            weights[i] = 0  
        elif depth_of_top_of_soil_level[i] <= required_depth_of_top_of_soil_level and \
             depth_of_base_of_soil_level[i] >= required_depth_of_base_of_soil_level: 
            # required level completely inside model level
            weights[i] = ( required_depth_of_base_of_soil_level 
                         - required_depth_of_top_of_soil_level ) / dz[i] 
        else: 
            # required level and model level overlap, but required level is not
            # completely inside model level 
            weights[i] = (
                         (min(depth_of_base_of_soil_level[i], required_depth_of_base_of_soil_level))
                         -
                         (max(depth_of_top_of_soil_level[i], required_depth_of_top_of_soil_level))
                         ) / dz[i]
        
    if np.sum(weights) <= 0.0:
        raise UserWarning('something has gone wrong here')
                
    return weights
    
    
def convert_vol_sm_to_water_content(vol_sm, dz_in_m): 
    # vol_sm is in m3 / m3
    # dz is in m
    # water content is in kg m-2
    
    if vol_sm.shape != dz_in_m.shape:
        raise UserWarning('vol_sm and dz_in_m should be the same shape')
        
    water_content = vol_sm * dz_in_m # m
    water_content *= 1000.0 # mm i.e. kg m-2
    
    return water_content
    
    
def convert_water_content_to_vol_sm(water_content, dz_in_m): 
    # water is in kg m-2   
    # dz is in m
    # vol_sm is in m3 / m3

    if water_content.shape != dz_in_m.shape:
        raise UserWarning('water_content and dz_in_m should be the same shape')
        
    vol_sm = water_content / dz_in_m # kg m-2 / m = mm / m
    vol_sm /= 1000.0

    return vol_sm
    
    
def get_vol_soil_moisture_cube_on_new_levels(cube_old, dz_old=None, dz_new=None):
    
    cubelist_new = iris.cube.CubeList([])
    
    old_depth_coord = cube_old.coord('depth')
    
    depth_of_top_of_soil_level = np.cumsum(dz_old) - dz_old
    depth_of_base_of_soil_level = np.cumsum(dz_old)
    
    if np.abs(np.sum(dz_old) - np.sum(dz_new)) > 1.0E-6:
        print(np.sum(dz_old), np.sum(dz_new))
        raise UserWarning('total soil column in dz_new is not the same as dz_old')
    
    if len(dz_old) != len(cube_old.coord('depth').points):
        raise UserWarning('dz_old is not the same length as the depth coord in cube_old')    
    
    for i,point in enumerate(cube_old.coord('depth').points):
        if not (depth_of_top_of_soil_level[i] <= point <= depth_of_base_of_soil_level[i]):
            print(i,depth_of_top_of_soil_level[i], point, depth_of_base_of_soil_level[i])
            raise UserWarning('dz_old is not consistent with the depth coord in cube_old')    
    
    required_depth_of_top_of_soil_level = np.cumsum(dz_new) - dz_new
    required_depth_of_base_of_soil_level = np.cumsum(dz_new)
    required_depth_of_middle_of_soil_level = (
        required_depth_of_top_of_soil_level + required_depth_of_base_of_soil_level
        ) / 2.0
    
    new_depth_coord = iris.coords.DimCoord(required_depth_of_middle_of_soil_level, var_name='depth',
        bounds = [[required_depth_of_top_of_soil_level[i], required_depth_of_base_of_soil_level[i]] 
            for i in range(len(dz_new))])
    
    var_name = cube_old.var_name
    
    for sl in cube_old.slices('depth'):
        arr_old = sl.data
        scalar_coords = sl.coords(dimensions=())
        
        arr_new = get_vol_soil_moisture_on_new_levels(arr_old, dz_old=dz_old, dz_new=dz_new)
        
        cube = iris.cube.Cube(arr_new, var_name=var_name)
        
        cube.add_aux_coord(new_depth_coord, (0,))
        for scalar_coord in scalar_coords:
            cube.add_aux_coord(scalar_coord)
        
        cubelist_new.append(cube)
        
    cube_new = cubelist_new.merge_cube()   
    
    return cube_new
    
    
def get_vol_soil_moisture_on_new_levels(vol_sm_old, dz_old=None, dz_new=None):
    # vol_sm_old and vol_sm_new are in m3 / m3
    # dz_old and dz_new should be in m
    
    water_content_old = convert_vol_sm_to_water_content(vol_sm_old, dz_in_m=dz_old)
    
    water_content_new = get_water_content_on_new_levels(water_content_old, 
        dz_old=dz_old, dz_new=dz_new)
        
    vol_sm_new = convert_water_content_to_vol_sm(water_content_new, dz_in_m=dz_new)
    
    return vol_sm_new
    
    
def get_water_content_on_new_levels(water_content_old, dz_old=None, dz_new=None):
    # water_content_old and water_content_new are in kg m-2
    # dz_old and dz_new should be in the same units
        
    required_depth_of_top_of_soil_level = np.cumsum(dz_new) - dz_new
    required_depth_of_base_of_soil_level = np.cumsum(dz_new)
    
    water_content_new = np.zeros(len(dz_new))
    
    for i in range(len(dz_new)):
        
        weights = calc_weights(dz=dz_old, 
            required_depth_of_top_of_soil_level=required_depth_of_top_of_soil_level[i], 
            required_depth_of_base_of_soil_level=required_depth_of_base_of_soil_level[i])
        
        water_content_new[i] = np.sum(weights * water_content_old)

        # check that no needed data is masked
        for j in range(len(weights)):
            if weights[j] > 0.0 and water_content_old[j] is np.ma.masked:
                water_content_new[i] = np.nan                 
            
    return water_content_new
    
    
def get_index_from_depth(depth, dz=None):

    index = None
    
    depth_of_top_of_soil_level = np.cumsum(dz) - dz
    depth_of_base_of_soil_level = np.cumsum(dz)
    
    for i in range(len(depth_of_top_of_soil_level)):
        if depth_of_top_of_soil_level[i] < depth <= depth_of_base_of_soil_level[i]:
            index = i
            break
            
    return index
  

def depths_from_var_name(var_name):
    '''
    Extract the depths from the var_name.
    e.g. 
       if var_name='vol_soil_water_at_10cm', then this function returns 0.1.
       if var_name='vol_soil_water_0-30cm', then this function returns [0.0, 0.3].
    depths are in m
    '''

    possible_var_names = [ 
        'vol_soil_water_0-30cm', 'vol_soil_water_30-60cm', 'vol_soil_water_60-90cm', 
        'vol_soil_water_90-120cm', 'vol_soil_water_120-150cm', 'vol_soil_water_150-180cm',
        'vol_soil_water_at_10cm',
        'vol_soil_water_at_30cm',
        'vol_soil_water_at_50cm',
        'vol_soil_water_at_90cm',
        'vol_soil_water_at_190cm',
        'soil_temperature_at_2cm',
        'soil_temperature_at_6cm',
        'soil_temperature_at_10cm',
        'soil_temperature_at_30cm',
        'soil_temperature_at_50cm',
        'soil_temperature_at_100cm',
        'soil_temperature_at_200cm'
        ]
            
    if var_name not in possible_var_names:
        raise UserWarning('This var_name has not been implemented')

    if '_at_' in var_name:
        retext = re.search("at_(?P<depth>[0-9\.]+)cm", var_name)

        res = float(retext.group('depth')) * 0.01 # convert cm to m

    else:
        retext = re.search("(?P<depth1>[0-9\.]+)-(?P<depth2>[0-9\.]+)cm", var_name)

        depth1 = float(retext.group('depth1')) * 0.01 # convert cm to m
        depth2 = float(retext.group('depth2')) * 0.01 # convert cm to m

        res = [depth1, depth2]

    return res
    

def lin_interp_layers(arr_1d, dz=None, pick_z=None, surface_val=None):

    if surface_val is not None:
        dz = np.insert(dz, 0, 0.0)
        arr_1d = np.insert(arr_1d, 0, surface_val)

    depth_of_base_of_soil_level = np.cumsum(dz)
    depth_of_top_of_soil_level = depth_of_base_of_soil_level - dz

    mid_points = 0.5 * (depth_of_base_of_soil_level + depth_of_top_of_soil_level)
   
    if ( pick_z < mid_points[0] ) or ( pick_z > mid_points[-1] ):
        raise UserWarning('this routine only works within the centres of each layer')

    val = np.interp(pick_z, mid_points, arr_1d)

    return val


def add_depth_coord(cube, existing_soil_coord_str, dz):
    '''
    Add a depth coord to a cube, using the midpoint of the soil levels
    '''

    depths = np.cumsum(dz) - 0.5 * dz
    
    cube.add_aux_coord(iris.coords.AuxCoord(depths, var_name='depth'), cube.coord_dims(existing_soil_coord_str))
    
    return
    
    
def main():
    dz = np.array([0.1, 0.25, 0.65, 2.0])
    
    thickness = 0.3
        
    for i in range(6):
        required_depth_of_top_of_soil_level = thickness * i
        required_depth_of_base_of_soil_level = thickness * ( i + 1 )
        
        weights = calc_weights(dz=dz, 
                               required_depth_of_top_of_soil_level=required_depth_of_top_of_soil_level, 
                               required_depth_of_base_of_soil_level=required_depth_of_base_of_soil_level)
                               
        print('~~~')                  
        print(required_depth_of_top_of_soil_level, 'to', required_depth_of_base_of_soil_level)
        print(weights)
        

if __name__ == '__main__':
    main()
