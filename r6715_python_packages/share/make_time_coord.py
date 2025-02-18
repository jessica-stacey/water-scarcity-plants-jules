# -*- coding: iso-8859-1 -*-
import datetime

import numpy as np
    
import iris   

if iris.__version__ != '1.8.1': # needed because monsoon postproc has 1.8.1 and no cf_units
    import cf_units

'''
Contains the function make_time_coord, which is useful when reading observational data 
used in the PEG into iris cubes.

Karina Williams
'''

def add_time_middle(cube, coord):
    '''
    Adds a coordinate that is the midpoint of the time bounds.
    Copied from KW's myutils.py 06/07/2020 (see myutils.py for notes on possible issues with this function)
    '''
    if coord.has_bounds():
        time_middle_aux_coord = coord.copy()
        time_middle_aux_coord.rename('time_middle') 
        
        points = (coord.bounds[:,0] + coord.bounds[:,1])/2
        time_middle_aux_coord.points = points

        cube.add_aux_coord(time_middle_aux_coord, cube.coord_dims(coord))


def wrap_unit(*args, **kwargs):
    '''
    Wraps the creation of a Unit object, so that the script can still run with the old version
    of Iris on the monsoon postprocessor.
    Will get rid of this function when Iris gets updated there.
    '''
    
    if iris.__version__ == '1.8.1':
        return iris.unit.Unit(*args, **kwargs)
    else:
        return cf_units.Unit(*args, **kwargs)


def create_time_coord_from_array_of_datetimes(datetime_arr):
    '''
    Creates an iris DimCoord from an array of datetimes 
    '''

    time_unit = wrap_unit('hours since epoch', 'gregorian')

    time_coord = iris.coords.DimCoord(
        time_unit.date2num(datetime_arr),
        standard_name='time',
        units=time_unit)

    return time_coord
    

def round_day_of_year_float_to_nearest_minute(day_of_year_float):
    '''
    Rounds the day of year (given as a float) to the nearest minute. Useful for e.g. US-SRM and US-SRG.
    '''

    minutes_in_day = 24.0 * 60.0
    rounded_day_of_year = day_of_year_float * minutes_in_day
    rounded_day_of_year = round(rounded_day_of_year)
    rounded_day_of_year /= minutes_in_day
    
    return rounded_day_of_year
    
    
def make_time_coord(data):
    '''
    Uses the information read in from the file to create an iris DimCoord containing the time.
    data is normally a structured array (e.g. read in from np.genfromtxt) but
    can also be a dictionary.
    '''

    if isinstance(data, dict):
        names = data.keys()        
    else:    
        names = data.dtype.names
        
    if 'TIMESTAMP' in names:
        time_coord = create_time_coord_from_array_of_datetimes(data['TIMESTAMP'])
    
    elif all(item in names for item in ['YEAR', 'DOY', 'HRMIN']):

        if np.max(data['HRMIN']) > 24.0:
            raise UserWarning('need to check the HRMIN column')

        if np.max(data['YEAR']) < 0:
            raise UserWarning('need to check the YEAR column')

        dt_array = []
        for i in range(len(data['YEAR'])):  
           dt = datetime.datetime(data['YEAR'][i],1,1) +  datetime.timedelta(days=data['DOY'][i] - 1) + datetime.timedelta(hours=data['HRMIN'][i])
           dt_array.append(dt)

        time_coord = create_time_coord_from_array_of_datetimes(dt_array)
    elif all(item in names for item in ['Year_LBAMIP', 'DoY_LBAMIP', 'Hour_LBAMIP']):
        dt_array = []
        for i in range(len(data['Year_LBAMIP'])):  
           dt = datetime.datetime(data['Year_LBAMIP'][i],1,1) +  datetime.timedelta(days=data['DoY_LBAMIP'][i] - 1) + datetime.timedelta(hours=data['Hour_LBAMIP'][i])
           dt_array.append(dt)

        time_coord = create_time_coord_from_array_of_datetimes(dt_array)
    elif all(item in names for item in ['Year_LBAMIP', 'DoY_LBAMIP']):
        dt_array = []
        for i in range(len(data['Year_LBAMIP'])):  
           dt = datetime.datetime(data['Year_LBAMIP'][i],1,1) +  datetime.timedelta(days=data['DoY_LBAMIP'][i] - 1) 
           dt_array.append(dt)

        time_coord = create_time_coord_from_array_of_datetimes(dt_array)
    elif all(item in names for item in ['Year', 'Month', 'Day_of_Month', 'Time_of_Day']): # e.g. SD-Dem (n.b. day_of_month in SD-Dem in 2012 is sometimes wrong )
        dt_array = []

        if set([data['Time_of_Day'][1] - data['Time_of_Day'][0], 
            data['Time_of_Day'][2] - data['Time_of_Day'][1]]) == set([30, 70]):
            n_timesteps_in_day = 48
        else:
            print data['Time_of_Day']
            raise UserWarning('need to code this case up')

        for i in range(len(data['Year'])): 
            if data['Year'][i] is np.ma.masked: # gap fill datetime              
               dt = dt_array[i - n_timesteps_in_day] + datetime.timedelta(days=1)
            else:
               time_of_day = data['Time_of_Day'][i] 
               minutes = int(str(time_of_day)[-2:])

               hours = 0.01 * (time_of_day - minutes) + minutes / 60.0
               hours -= 0.25 #so that first timestep of day is 0.25, and last is 23.75  
         
               dt = datetime.datetime(data['Year'][i], data['Month'][i], data['Day_of_Month'][i]) \
                   + datetime.timedelta(hours=hours) 
               #dt = datetime.datetime(data['Year'][i], 1, 1) \
               #    + datetime.timedelta(days=data['Julian_Day'][i] - 1) \
               #    + datetime.timedelta(hours=hours) 
            dt_array.append(dt)

        #for i in range(1,len(dt_array)):
        #    if dt_array[i] - dt_array[i-1] != datetime.timedelta(minutes=30):
        #        print dt_array[i], dt_array[i-1], dt_array[i] - dt_array[i-1]

        time_coord = create_time_coord_from_array_of_datetimes(dt_array)

    elif all(item in names for item in ['Year', 'Julian_Day']): # e.g. US-SRM, US-SRG
        dt_array = []
        for i in range(len(data['Year'])):  
           day_of_year = round_day_of_year_float_to_nearest_minute(data['Julian_Day'][i])
           dt = datetime.datetime(int(data['Year'][i]), 1, 1) +  datetime.timedelta(days=day_of_year - 1.0) 
           dt_array.append(dt)

        time_coord = create_time_coord_from_array_of_datetimes(dt_array)

    elif 'Date_k67' in names:
    
        dt_format = '%m/%d/%Y'
        dt_array = [datetime.datetime.strptime(x, dt_format) for x in data['Date_k67']]
        print dt_array
        time_coord = create_time_coord_from_array_of_datetimes(dt_array)
        print time_coord
    else:  
        raise Exception('not enough information to create time coord')
    return time_coord
    

def convert_calendar_from_365_day_to_gregorian(cube):
    '''
    Converts time coord on cube from 365_day calendar to gregorian.
    Also gets rid of any bounds.
    '''
    if not cube.coords('time'):
        raise Exception('cube does not have a time coord')
        
    old_time_coord = cube.coord('time').copy()
    
    if old_time_coord.units.calendar != '365_day':
        raise Exception('function is expecting calendar in input cube to be 365_day')

    dt_array = old_time_coord.units.num2date(cube.coord('time').points)
       
    new_time_coord = create_time_coord_from_array_of_datetimes(dt_array)
    
    if new_time_coord.units.calendar != 'gregorian':
        raise Exception('function is expecting calendar in new time coord to be gregorian')
    
    time_coord_dims = cube.coord_dims('time')
    
    cube.remove_coord('time')
    
    cube.add_dim_coord(new_time_coord, time_coord_dims)
        
    for att in ['var_name', 'long_name']:     
        if hasattr(old_time_coord, att):
            setattr(new_time_coord, att, getattr(old_time_coord, att))
        
    return cube
    

def interpolate_to_get_daily_cube(in_cube):
    '''
    Returns a cube which has been interpolated to daily resolution.
    Requires an input cube at daily resolution or lower (e.g. the 16 day LBA files) - just 
    checks the separation of the first two time points to determine this.
    '''
    
    time_unit = in_cube.coord('time').units

    one_day = time_unit.date2num(datetime.datetime(2000,1,2)) \
            - time_unit.date2num(datetime.datetime(2000,1,1))

    if one_day > in_cube.coord('time').points[1] - in_cube.coord('time').points[0]:
        raise UserWarning('do not want to use this function if the timestep is less than a day')

    new_points_array = np.arange(in_cube.coord('time').points[0], 
        in_cube.coord('time').points[-1] + one_day, one_day)
    out_cube = in_cube.interpolate([('time', new_points_array)], iris.analysis.Linear())
   
    if out_cube.coord('time')[-1] != in_cube.coord('time')[-1]:
        raise UserWarning('Something has gone wrong here. E.g. Check whether there is daylight saving.')
    
    return out_cube
