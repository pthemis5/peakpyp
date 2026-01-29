#### DELETE ME PLEASE
import numpy as np
PI = np.pi
# firstly we will define JD at 1 January 2025
jd2025 = 2460677.0
# we choose this value, because we do not want to bother with disekta years

def month_days(year, month):
    """
    This function will return the number of days in a given month
    """
    if month == 2:
        if year % 4 == 0:
            return 29
        else:
            return 28
    elif month in [4, 6, 9, 11]:
        return 30
    else:
        return 31
    


def sum_month_dates(year, month):
    """
    This function will return the sum of days in all months up to the given month
    """
    return sum([month_days(year, i) for i in range(1, month)])

    

 ######### cides bekiw can be simplfied, if we decide not to bother about leap years 
def year_days(year):
    """
    This function will return the number of days in a given year
    """
    if year % 4 == 0:
        return 366
    else:
        return 365
    

def sum_year_dates(year):
    """
    This function will return the sum of days in all years up to the given year
    """
    ## we can avoid these mins or maxes, if we just decide that the year will be more than 2025
    if year > 2025:
        return sum([year_days(i) for i in range(min(2025, year), max(year, 2025))])
    
    elif year < 2025:
        return - sum([year_days(i) for i in range(min(2025, year), max(year, 2025))])
    
    else:
        return 0
    


def get_jd(year, month, day, hour, minute, second):
    """
    This function will return the Julian Date of the given date

    Parameters
    ----------
    all are ints, appart from seconds that can have decimal places


    Returns
    -------
    float
        Julian Date
    """

    dhr = hour
    dmi = minute
    dsc = second

    year_days_from_2025 = sum_year_dates(year)
    month_days = sum_month_dates(year, month)

    
    return year_days_from_2025 + month_days + (day - 1.5) + dhr/24 + dmi/1440 + dsc/86400 + jd2025



def gst_from_utc(jd):
    T = (jd - 2451545.0) / 36525
    GST = 280.46061837 + 360.98564736629 * (jd - 2451545) + 0.000387933 * T**2 - (T**3) / 38710000.0
    GST = np.radians(GST % 360)
    return GST


def z_rot(theta, x, y, z):
    # Rotate a point (x, y, z) by an angle theta around the z-axis
    x_new = x * np.cos(theta) - y * np.sin(theta)
    y_new = x * np.sin(theta) + y * np.cos(theta)
    z_new = z
    return x_new, y_new, z_new



def ecef_to_eci_stationary(ecef_coords, gst):
    # Convert ECEF coordinates to ECI
    X_ecef, Y_ecef, Z_ecef = ecef_coords
    X_eci, Y_eci, Z_eci = z_rot(gst, X_ecef, Y_ecef, Z_ecef)
    return np.array([X_eci, Y_eci, Z_eci])

def eci_to_ecef_stationary(ecef_coords, gst):
    # Convert ECEF coordinates to ECI
    X_ecef, Y_ecef, Z_ecef = ecef_coords
    X_eci, Y_eci, Z_eci = z_rot(-gst, X_ecef, Y_ecef, Z_ecef)
    return np.array([X_eci, Y_eci, Z_eci])

def geodetic_to_ecef(lat, lon, alt, a, e_sq):
    # Convert geodetic coordinates to ECEF
    lat = np.radians(lat)
    lon = np.radians(lon)
    
    N = a / np.sqrt(1 - e_sq* np.sin(lat)**2)
    
    X = (N + alt) * np.cos(lat) * np.cos(lon)
    Y = (N + alt) * np.cos(lat) * np.sin(lon)
    Z = (N * (1 - e_sq) + alt) * np.sin(lat)
    
    return np.array([X, Y, Z])


def get_theta(vec1_array, vec2_array):
    # intprod = np.sum(vec1_array * vec2_array, axis = 1) / np.sqrt(np.sum(hopefully_directions**2, axis = 1)) / np.sqrt(np.sum(normalized_direction**2, axis = 1))
    intprod = np.sum(vec1_array * vec2_array, axis = 1) / np.linalg.norm(vec1_array, axis = 1) / np.linalg.norm(vec2_array, axis = 1)
    return np.arccos(intprod) * 180 / PI


