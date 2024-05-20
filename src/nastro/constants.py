"""
Constants
==========================================

Physical and mathematical constants and units taken from SciPy.

======================================= ===============================================
``pi``, ``twopi``, ``halfpi``           Pi and multiples
``c``                                   speed of light in vacuum
``G``                                   Newtonian constant of gravitation
``degree``                              degree in radians
``arcmin``                              arc minute in radians
``arcsec``                              arc second in radians
``miliarcsec``                          milliarcsecond in radians
``minute``                              one minute in seconds
``hour``                                one hour in seconds
``day``                                 one day in seconds
``year``                                one year (365 days) in seconds
``julian_year``                         one Julian year (365.25 days) in seconds
``au``                                  one astronomical unit in meters
``light_year``                          one light year in meters
``parsec``                              one parsec in meters
``km``                                  kilometers in meters
``kmh``                                 kilometers per hour in meters per second
======================================= ===============================================
"""

import scipy.constants as sc

# Mathematical constants
pi = sc.pi
twopi = 2.0 * pi
halfpi = 0.5 * pi

# Physical constants
c = clight = sc.speed_of_light
G = sc.gravitational_constant

# Angles in radians
degree = pi / 180.0
arcmin = degree / 60.0
arcsec = arcmin / 60.0
miliarcsec = arcsec * 1e-3

# Time in seconds
minute = 60.0
hour = minute * 60.0
day = hour * 24.0
year = day * 365
julian_year = day * 365.25

# Lengths in meters
km = 1e3
au = astronomical_unit = sc.astronomical_unit
light_year = sc.light_year
parsec = sc.parsec
km = 1e3

# Speed in m/s
kmh = 1e3 / hour
