"""Define a ROMS grid file, and fill in the map values.
This is derived from grid.py in the repo
https://github.com/bjornaa/gridmap
by:

Bjørn Ådlandsvik <bjorn@imr.no>

Institute of Marine Research

2012-04-15
Changes by Chris Sherwood, USGS to write generic lat/lon and x,y data to a netcdf file
"""
import numpy as np
from netCDF4 import Dataset
from datetime import datetime

def create_roms_netcdf_grid_file(grid_name, Lm, Mm,
                        global_attributes, format):
    """
    Create a new empty ROMS grid file
    Arguments:
      grid_name : name of the grid (filename will be grid_name.nc)
      format    : 'NETCDF3_CLASSIC' or 'NETCDF4_CLASSIC' or 'NETCDF4'

    Make space for all variables, including coordinate variables
    This tries to replicate behavior of create_roms_netcdf_grid_file.m
    Modified by Chris Sherwood
    """

    file_name = grid_name + '.nc'
    ds = '{:%B %d, %Y}'.format(datetime.now())
    print('New',ds)
    print('Generating ',file_name)

    # # Ta denne med polarstereografisk ???
    # gridmap_varname = 'grid_mapping' # Name of grid mapping variable

    # -----------------------
    # NetCDF file definition
    # -----------------------

    nc = Dataset(file_name, 'w', format=format)

    # Global attributes
    # Defaults
    global_defaults = dict(gridname    = grid_name,
                           type        = 'ROMS GRID file',
                           history     = 'Created by create_roms_netcdf_grid_file.py in grid.py on '+ds,
                           title       = 'ROMS application',
                           Conventions = 'CF-1.2')
    d = {}
    d.update(global_defaults, **global_attributes)

    for att, value in d.items():
        setattr(nc, att, value)

    # Dimensions
    L,  M  = Lm+1, Mm+1
    Lp, Mp = Lm+2, Mm+2
    one = 1
    two = 2
    nc.createDimension('xi_psi',  L)
    nc.createDimension('xi_rho',  Lp)
    nc.createDimension('xi_u',    L)
    nc.createDimension('xi_v',    Lp)
    nc.createDimension('eta_psi', M)
    nc.createDimension('eta_rho', Mp)
    nc.createDimension('eta_u',   Mp)
    nc.createDimension('eta_v',   M)
    nc.createDimension('maxStrlen64', 64)
    nc.createDimension('one',     1)
    nc.createDimension('two',     2)
    nc.createDimension('bath',    one)

    # --- Coordinate variables ---
    # Not required by ROMS, but recommended by the CF standard
    v1 = nc.createVariable('xl', 'd', ('one',) )
    v1.long_name = "domain length in the XI-direction"
    v1.units = "meter"

    v2 = nc.createVariable('el', 'd', ('one',) )
    v2.long_name = "domain length in the ETA-direction"
    v2.units = "meter"

    v3 = nc.createVariable('JPRJ', 'c', ('maxStrlen64',))
    v3.long_name = "Map projection type"
    v3.flag_values = "option_ME, option_ST, option_LC"
    v3.flag_meanings = "Mercator, Sterographic, Lambert conformal conic"

    v4 = nc.createVariable('spherical', 'c', ('maxStrlen64',))
    v4.long_name = "Grid type logical switch"
    v4.flag_values = "T, F"
    v4.flag_meanings = "spherical Cartesian"

    v5 = nc.createVariable('depthmin', 'f', ('one',))
    v5.long_name = "Minimum depth"
    v5.units = "meter"

    v6 = nc.createVariable('depthmax', 'f', ('one',))
    v6.long_name = "Deep bathymetry clipping depth"
    v6.units = "meter"

    # --- Topography
    v7 = nc.createVariable('hraw', 'd', ('bath', 'eta_rho', 'xi_rho'), zlib=True)
    v7.long_name = "Working bathymetry at RHO-points"
    v7.standard_name = "sea_floor_depth"
    v7.units = "meter"
    v7.field = "bath, scalar"
    v7.coordinates = "lon_rho lat_rho"

    v8 = nc.createVariable('h', 'd', ('eta_rho', 'xi_rho'), zlib=True)
    v8.long_name = "Final bathymetry at RHO-points"
    v8.standard_name = "sea_floor_depth"
    v8.units = "meter"
    v8.field = "bath, scalar"
    v8.coordinates = "lon_rho lat_rho"

    # --- Coriolis
    v9 = nc.createVariable('f', 'd', ('eta_rho', 'xi_rho'), zlib=True)
    v9.long_name = 'Coriolis parameter at RHO-points'
    v9.standard_name = "coriolis_parameter"
    v9.units = 'second-1'
    v9.field = 'Coriolis, scalar'
    v9.coordinates = "lon_rho lat_rho"

    # --- Metric terms
    v10 = nc.createVariable('pm', 'd', ('eta_rho', 'xi_rho'), zlib=True)
    v10.long_name = "curvilinear coordinate metric in XI"
    v10.units = "meter-1"
    v10.field = "pm, scalar"
    v10.coordinates = "lon_rho lat_rho"

    v11 = nc.createVariable('pn', 'd', ('eta_rho', 'xi_rho'), zlib=True)
    v11.long_name = "curvilinear coordinate metric in ETA"
    v11.units = "meter-1"
    v11.field = "pn, scalar"
    v11.coordinates = "lon_rho lat_rho"

    v12 = nc.createVariable('dndx', 'd', ('eta_rho', 'xi_rho'), zlib=True)
    v12.long_name = "xi derivative of inverse metric factor pn"
    v12.units = "meter"
    v12.field = "dndx, scalar"
    v12.coordinates = "lon_rho lat_rho"

    v13 = nc.createVariable('dmde', 'd', ('eta_rho', 'xi_rho'), zlib=True)
    v13.long_name = "eta derivative of inverse metric factor pm"
    v13.units = "meter"
    v13.field = "pn, scalar"
    v13.coordinates = "lon_rho lat_rho"

    v14 = nc.createVariable('x_rho', 'd', ('eta_rho','xi_rho'), zlib=True)
    v14.long_name = "X coordinate of RHO-points"
    v14.standard_name = "projection_x_coordinate"
    v14.units = "meter"

    v15 = nc.createVariable('y_rho', 'd', ('eta_rho','xi_rho'), zlib=True)
    v15.long_name = "Y coordinate of RHO-points"
    v15.standard_name = "projection_y_coordinate"
    v15.units = "meter"

    v16 = nc.createVariable('x_psi', 'd', ('eta_psi','xi_psi'), zlib=True)
    v16.long_name = "X coordinate of PSI-points"
    v16.units = "meter"

    v17 = nc.createVariable('y_psi', 'd', ('eta_psi','xi_psi'), zlib=True)
    v17.long_name = "Y coordinate of PSI-points"
    v17.units = "meter"

    v18 = nc.createVariable('x_u', 'd', ('eta_u','xi_u',), zlib=True)
    v18.long_name = "X coordinate of U-points"
    v18.units = "meter"

    v19 = nc.createVariable('y_u', 'd', ('eta_u','xi_u'), zlib=True)
    v19.long_name = "Y coordinate of U-points"
    v19.units = "meter"

    v20 = nc.createVariable('x_v', 'd', ('eta_u','xi_v',), zlib=True)
    v20.long_name = "X coordinate of V-points"
    v20.units = "meter"

    v21 = nc.createVariable('y_v', 'd', ('eta_v','xi_v'), zlib=True)
    v21.long_name = "Y coordinate of V-points"
    v21.units = "meter"

    # --- Geographic variables
    v22 = nc.createVariable('lat_rho', 'd', ('eta_rho', 'xi_rho'), zlib=True)
    v22.long_name = "latitude of RHO-points"
    v22.standard_name = "latitude"
    v22.units = "degrees_north"

    v23 = nc.createVariable('lon_rho', 'd', ('eta_rho', 'xi_rho'), zlib=True)
    v23.long_name = "longitude of RHO-points"
    v23.standard_name = "longitude"
    v23.units = "degrees_east"

    v24 = nc.createVariable('lat_psi', 'd', ('eta_psi', 'xi_psi'), zlib=True)
    v24.long_name = "longitude of U-points"
    v24.standard_name = "longitude"
    v24.units = "degrees_east"

    v25 = nc.createVariable('lon_psi', 'd', ('eta_psi', 'xi_psi'), zlib=True)
    v25.long_name = "longitude of U-points"
    v25.standard_name = "longitude"
    v25.units = "degrees_east"

    v26 = nc.createVariable('lat_u', 'd', ('eta_u', 'xi_u'), zlib=True)
    v26.long_name = "latitude of U-points"
    v26.standard_name = "latitude"
    v26.units = "degrees_north"

    v27 = nc.createVariable('lon_u', 'd', ('eta_u', 'xi_u'), zlib=True)
    v27.long_name = "longitude of U-points"
    v27.standard_name = "longitude"
    v27.units = "degrees_east"

    v28 = nc.createVariable('lat_v', 'd', ('eta_v', 'xi_v'), zlib=True)
    v28.long_name = "latitude of V-points"
    v28.standard_name = "latitude"
    v28.units = "degrees_north"

    v29 = nc.createVariable('lon_v', 'd', ('eta_v', 'xi_v'), zlib=True)
    v29.long_name = "longitude of V-points"
    v29.standard_name = "longitude"
    v29.units = "degrees_east"

    # --- Masks
    v30 = nc.createVariable('mask_rho', 'd', ('eta_rho', 'xi_rho'), zlib=True)
    v30.long_name = "mask on RHO-points"
    #v.standard_name = "sea_binary_mask"   # Not in standard table
    v30.option_0 = "land"
    v30.option_1 = "water"
    v30.coordinates = "lon_rho lat_rho"

    v31 = nc.createVariable ('mask_u', 'd', ('eta_u', 'xi_u'), zlib=True)
    v31.long_name = "mask on U-points"
    v31.option_0 = "land"
    v31.option_1 = "water"
    v31.coordinates = "lon_u lat_u"

    v32 = nc.createVariable('mask_v', 'd', ('eta_v', 'xi_v'), zlib=True)
    v32.long_name = "mask on V-points"
    v32.option_0 = "land"
    v32.option_1 = "water"
    v32.coordinates = "lon_v lat_v"

    v33 = nc.createVariable('mask_psi', 'd', ('eta_psi', 'xi_psi'), zlib=True)
    v33.long_name = "mask on PSI-points"
    v33.option_0 = "land"
    v33.option_1 = "water"

    v34 = nc.createVariable('angle', 'd', ('eta_rho', 'xi_rho'), zlib=True)
    v34.long_name = "angle between xi axis and east"
    v34.standard_name = "angle_of_rotation_from_east_to_x"
    v34.units = "degrees_east"
    v34.coordinates = "lon_rho lat_rho"

    nc.close()


def create_grid(gmap, grid_name, file_name='',
                global_attributes={},
                format='NETCDF3_CLASSIC'):
    """
    Create a new ROMS grid file for a polar stereographic grid
    Arguments:
      gmap      : a gridmap.PolarStereographic instance
      grid_name : name of the grid
      file_name : name of the grid file,
                  default = '' giving grid_name + '_grid.nc'
      format    : 'NETCDF3_CLASSIC' or 'NETCDF4_CLASSIC'
                  default = 'NETCDF3_CLASSIC'
    Fills in geometric variables (lons, lats, metric, Coriolis).
    Makes space for topographic variables (h, hraw, masks).
    Also makes coordinate variables and includes grid mapping info
    following the CF-standard.

    """

    if not file_name:  # Use default
        file_name = grid_name + '_grid.nc'

    make_empty_gridfile(grid_name, file_name, gmap.Lm, gmap.Mm,
                        global_attributes=global_attributes,
                        format=format)

    nc = Dataset(file_name, 'a')

    gridmap_varname = 'grid_mapping' # Name of grid mapping variable
    Lm, Mm = gmap.Lm, gmap.Mm

    # --- Grid map

    #v = nc.createVariable(gridmap_varname, 'i', ())
    v = nc.variables[gridmap_varname]
    #v.long_name = "grid mapping"
    d = gmap.CFmapping_dict()
    for att in d:
        setattr(v, att, d[att])
    v.proj4string = gmap.proj4string


    # ------------------------------------------------------
    # Compute variables defined by only by the grid mapping
    # ------------------------------------------------------

    #print "Saving geometric variables"

    # -----------------------
    # Coordinate variables
    # -----------------------
    nc.variables['xi_rho'][:]  = gmap.dx*np.arange(Lm+2)
    nc.variables['eta_rho'][:] = gmap.dx*np.arange(Mm+2)
    nc.variables['xi_u'][:]    = gmap.dx*(np.arange(Lm+1)+0.5)
    nc.variables['eta_u'][:]   = gmap.dx*np.arange(Mm+2)
    nc.variables['xi_v'][:]    = gmap.dx*np.arange(Lm+2)
    nc.variables['eta_v'][:]   = gmap.dx*(np.arange(Mm+1)+0.5)

    # ----------
    # Vertices
    # ----------

    # Vertices at every half point in the grid
    # -0.5, 0, 0.5, ...., Lm+1.5
    Lvert = 2*Lm + 5
    Mvert = 2*Mm + 5
    X0 = 0.5*np.arange(Lvert)-0.5
    Y0 = 0.5*np.arange(Mvert)-0.5
    # Make 2D arrays with grid coordonates
    Xvert, Yvert = np.meshgrid(X0, Y0)
    Xrho = Xvert[1::2, 1::2]
    Yrho = Yvert[1::2, 1::2]

    lon_vert, lat_vert = gmap.grid2ll(Xvert, Yvert)

    # Set the different points
    nc.variables['lon_rho'][:,:] = lon_vert[1::2, 1::2]
    lat_rho = lat_vert[1::2, 1::2]
    nc.variables['lat_rho'][:,:] = lat_rho
    nc.variables['lon_u'][:,:]   = lon_vert[1::2, 2:-1:2]
    nc.variables['lat_u'][:,:]   = lat_vert[1::2, 2:-1:2]
    nc.variables['lon_v'][:,:]   = lon_vert[2:-1:2, 1::2]
    nc.variables['lat_v'][:,:]   = lat_vert[2:-1:2, 1::2]

    # ----------------------
    # Metric coefficients
    # ----------------------

    #pm = 1.0 / (gmap.map_scale(Xrho, Yrho) * gmap.dx)
    pm = gmap.map_scale(Xrho, Yrho) / gmap.dx
    pn = pm
    nc.variables['pm'][:,:] = pm
    nc.variables['pn'][:,:] = pn

    # Alternative:
    # Could define pm and pn by differencing on the ellipsoid
    # However, for WGS84 the distance formula is complicated

    # --- Derivatives of metric coefficients

    # Use differencing, as the calculus is complicated for WGS84
    # the pm and pn fields are changing slowly and no
    # problems near the North Pole

    dndx = np.zeros_like(pm)
    dmde = np.zeros_like(pm)

    # Central differences
    dndx[:, 1:-1] = 0.5/pn[:, 2:] - 0.5/pn[:, :-2]
    dmde[1:-1, :] = 0.5/pm[2:, :] - 0.5/pm[:-2, :]

    # linear extrapolation to boundary
    dndx[:,0]  = 2*dndx[:,1]  - dndx[:,2]
    dndx[:,-1] = 2*dndx[:,-2] - dndx[:,-3]
    dmde[0,:]  = 2*dmde[1,:]  - dmde[2,:]
    dmde[-1,:] = 2*dmde[-2,:] - dmde[-3,:]

    # Alternative for spherical earth
    #phi0 = gmap.lat_ts*np.pi/180.0
    #R = gmap.ellipsoid.a
    #dndx = - (Xrho - gmap.xp)*gmap.dx / (pn**2 * R**2 * (1+np.sin(phi0)))
    #dmde = - (Yrho - gmap.yp)*gmap.dx / (pm**2 * R**2 * (1+np.sin(phi0)))

    # save the coefficients
    nc.variables['dndx'][:,:] = dndx
    nc.variables['dmde'][:,:] = dmde

    # ---------
    # Coriolis
    # ---------

    Aomega = 2 * np.pi * (1+1/365.24) / 86400 # earth rotation
    nc.variables['f'][:,:] = 2 * Aomega * np.sin(lat_rho*np.pi/180.0)

    # ----------------
    # Rotation angle
    # ----------------

    nc.variables['angle'][:,:] = gmap.angle(Xrho, Yrho)
    # Could also be computed by differencing,
    # this would be very inaccurate near the North Pole

    # ------------------
    # Misc. variables
    # ------------------

    nc.variables['spherical'].assignValue('T')
    nc.variables['xl'].assignValue((Lm+1)*gmap.dx)
    nc.variables['el'].assignValue((Mm+1)*gmap.dx)

    # ---------------------
    # Close the grid file
    # ---------------------

    nc.close()

# -------------------------------------------------------

def subgridfile(file0, file1, i0, j0, Lm, Mm):
    ### Funker ikke helt

    f0 = Dataset(file0)
    gmap0 = gridmap.fromfile(f0)

    gmap1 = gridmap.subgrid(gmap0, i0, j0, Lm, Mm)

    grid_name = f0.gridname + "_sub"

    gridmap_varname = "grid_mapping"  # Les denne fra filen

    # Make an empty grid file of the correct shape
    make_empty_gridfile(grid_name, file1, Lm, Mm,
                        global_attributes={},format=f0.file_format)

    # Open this grid file
    f1 = Dataset(file1, 'a')
    # Add grid mapping
    v = f1.variables[gridmap_varname]
    #v.long_name = "grid mapping"
    d = gmap1.CFmapping_dict()
    for att in d:
        setattr(v, att, d[att])
    v.proj4string = gmap1.proj4string

    # -----------------------
    # Coordinate variables
    # -----------------------
    f1.variables['xi_rho'][:]  = gmap1.dx*np.arange(Lm+2)
    f1.variables['eta_rho'][:] = gmap1.dx*np.arange(Mm+2)
    f1.variables['xi_u'][:]    = gmap1.dx*(np.arange(Lm+1)+0.5)
    f1.variables['eta_u'][:]   = gmap1.dx*np.arange(Mm+2)
    f1.variables['xi_v'][:]    = gmap1.dx*np.arange(Lm+2)
    f1.variables['eta_v'][:]   = gmap1.dx*(np.arange(Mm+1)+0.5)

    # -------------------------
    # rho-point variables
    # ------------------------

    vars = ['lon_rho', 'lat_rho', 'mask_rho',
            'pm', 'pn', 'dmde', 'dndx', 'angle',
            'f', 'h']

    for var in vars:
        v0 = f0.variables[var]
        v1 = f1.variables[var]
        v1[:,:] = v0[j0:j0+Mm+2, i0:i0+Lm+2]

    # hraw is special
    v0 = f0.variables['hraw']
    v1 = f1.variables['hraw']
    for t in range(len(f0.dimensions['bath'])):
        v1[t,:,:] = v0[j0:j0+Mm+2, i0:i0+Lm+2]

    # u-point variables
    vars = ['lon_u', 'lat_u', 'mask_u']
    for var in vars:
        v0 = f0.variables[var]
        v1 = f1.variables[var]
        # Eller er det i0+1:i0+Lm+2
        v1[:,:] = v0[j0:j0+Mm+2, i0:i0+Lm+1]

    # v-point variables
    vars = ['lon_v', 'lat_v', 'mask_v']
    for var in vars:
        v0 = f0.variables[var]
        v1 = f1.variables[var]
        v1[:,:] = v0[j0:j0+Mm+1, i0:i0+Lm+2]

    # psi-point variables
    vars = ['mask_psi']
    for var in vars:
        v0 = f0.variables[var]
        v1 = f1.variables[var]
        # sjekk om indeksering er forskjøvet
        v1[:,:] = v0[j0:j0+Mm+1, i0:i0+Lm+1]

    # Some special variables
    f1.variables['spherical'].assignValue('T')
    f1.variables['xl'].assignValue((Lm+1)*gmap1.dx)
    f1.variables['el'].assignValue((Mm+1)*gmap1.dx)

    f1.close()

# some functions used here
def pcoord(x, y):
    """
    Convert x, y to polar coordinates r, az (geographic convention)
    r,az = pcoord(x, y)
    """
    r  = np.sqrt( x**2 + y**2 )
    az = np.degrees( np.arctan2(x, y) )
    # az[where(az<0.)[0]] += 360.
    az = (az+360.)%360.
    return r, az

def xycoord(r, az):
    """
    Convert r, az [degrees, geographic convention] to rectangular coordinates
    x,y = xycoord(r, az)
    """
    x = r * np.sin(np.radians(az))
    y = r * np.cos(np.radians(az))
    return x, y

def buildGrid(xp,yp,alp,dx,dy,mxc,myc):
    '''
    X, Y = buildGrid(xp,yp,alp,dx,dy,mxc,myc) builds a rectangular grid

    Input :
        xp: x grid origin
        yp: y grid origin
        alp: degrees rotation of x-axis
        dx: x-direction grid spacing
        dy: y-direction grid spacing
        mxc: number of meshes in x-direction
        myc: number of meshes in y-direction

    Based on the Matlab function of Dave Thompson
    '''
    xlen = (mxc-1)*dx
    ylen = (myc-1)*dy

    x = np.arange(xp,xp+xlen+dx,dx)
    y = np.arange(yp,yp+ylen+dy,dy)

    X,Y = np.meshgrid(x,y);
    X = X-xp
    Y = Y-yp

    if alp != 0.:
       r,az = pcoord(X,Y)
       X,Y = xycoord(r,az+alp)

    X = X+xp
    Y = Y+yp
    print("Shape of X and Y: ",np.shape(X),np.shape(Y))

    return X, Y
def buildROMSGrid(xp,yp,alp,dx,dy,Lm,Mm):
    '''
    X, Y = buildROMSGrid(xp,yp,alp,dx,dy,Lm,Mm) builds a rectangular grid

    Input :
        xp: x grid origin
        yp: y grid origin
        alp: degrees rotation of x-axis
        dx: x-direction grid spacing
        dy: y-direction grid spacing
        Lm: number of rho-points inside domain in x-direction
        Mm: number of rho-points inside domains in y-direction

    Based on the Matlab function of Dave Thompson
    '''
    L, M = Lm+1, Mm+1
    xlen = (Lm-1)*dx
    ylen = (Mm-1)*dy

    # Extend the grid one cell in each direction
    x = np.arange(xp-dx,xp+xlen+2.*dx,dx)
    y = np.arange(yp-dy,yp+ylen+2.*dy,dy)

    xu = np.arange(x[0]+dx/2.,x[-2]+dx/2,dx)
    print(np)
    X,Y = np.meshgrid(x,y);
    X = X-xp
    Y = Y-yp

    if alp != 0.:
       r,az = pcoord(X,Y)
       X,Y = xycoord(r,az+alp)

    X = X+xp
    Y = Y+yp
    print("Shape of X and Y: ",np.shape(X),np.shape(Y))

    return X, Y
