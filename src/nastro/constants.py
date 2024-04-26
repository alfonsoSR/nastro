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

# Speed in m/s
kmh = 1e3 / hour
