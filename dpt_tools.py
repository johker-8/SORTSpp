"""Functions from Daniel's-Python-tools package

This is a module to enable simple plotting with kwargs as
configuration and includes coordinate transformations and other useful functions.

# TODO: Fix so orbits work with hyperbolic
# TODO: Fix the 0-e 0-i errors
"""

import math
import time
import pdb
import os

os.environ['TZ'] = 'GMT'
time.tzset()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import scipy.constants
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations
import plothelp

M_earth = 398600.5e9/scipy.constants.G
'''float: Mass of the Earth using the WGS84 convention.
'''

M_sol = 1.98847e30 #kg
"""float: The mass of the sun :math:`M_\odot` given in kg, used in kepler transformations
"""


def hist2d(x, y, **options):
    """This function creates a histogram plot with lots of nice pre-configured settings unless they are overridden

    :param numpy.ndarray x:  data to histogram over, if x is not a vector it is flattened
    :param dict options: All keyword arguments as a dictionary containing all the optional settings.

    Currently the keyword arguments that can be used in the **options**:
        :bins [int]: the number of bins
        :colormap [str]: Name of colormap to use.
        :title [string]: the title of the plot
        :title_font_size [int]: the title font size
        :xlabel [string]: the label for the x axis
        :ylabel [string]: the label for the y axis
        :tick_font_size [int]: the axis tick font size
        :window [tuple/list]: the size of the plot window in pixels (assuming dpi = 80)
        :save [string]: will not display figure and will instead save it to this path
        :show [bool]: if False will do draw() instead of show() allowing script to continue
        :plot [tuple]: A tuple with the :code:`(fig, ax)` objects from matplotlib. Then no new figure and axis will be created.
        :logx [bool]: Determines if x-axis should be the logarithmic.
        :logy [bool]: Determines if y-axis should be the logarithmic.
        :log_freq [bool]: Determines if frequency should be the logarithmic.

    Example::
        
        import dpt_tools as dpt
        import numpy as np
        np.random.seed(19680221)

        x = 10*np.random.randn(1000)

        dpt.hist(x,
           title = "My first plot",
        )
        

    """

    if not isinstance(x, np.ndarray):
        _x = np.array(x)
    else:
        _x = x.copy()

    if _x.size != _x.shape[0]:
        _x = _x.flatten()

    if options.setdefault('logx', False ):
        _x = np.log10(_x)

    if not isinstance(y, np.ndarray):
        _y = np.array(y)
    else:
        _y = y.copy()

    if _y.size != _y.shape[0]:
        _y = _y.flatten()

    if options.setdefault('logy', False ):
        _y = np.log10(_y)


    #turn on TeX interperter
    plt.rc('text', usetex=True)

    if 'window' in options:
        size_in = options['window']
        size_in = tuple(sz/80.0 for sz in size_in)
    else:
        size_in=(15, 7)
    if 'bin_size' in options:
        options['bins'] = int(np.round((np.max(_x)-np.min(_x))/options['bin_size']))

    if 'plot' in options:
        fig, ax = options['plot']
    else:
        fig, ax = plt.subplots(figsize=size_in,dpi=80)

    cmap = getattr(cm, options.setdefault('cmap', 'plasma'))

    if options.setdefault('log_freq', False ):
        hst = ax.hist2d(_x, _y,
            bins=options.setdefault('bins', int(np.round(np.sqrt(_x.size))) ),
            cmap=cmap,
            norm=mpl.colors.LogNorm(),
        )
        ax.set_facecolor(cmap.colors[0])
    else:
        hst = ax.hist2d(_x, _y,
            bins=options.setdefault('bins', int(np.round(np.sqrt(_x.size))) ),
            cmap=cmap,
            normed=options.setdefault('pdf', False),
        )
    title = options.setdefault('title','Data scatter plot')
    if title is not None:
        ax.set_title(title,
            fontsize=options.setdefault('title_font_size',24))
    ax.set_xlabel(options.setdefault('xlabel','X-axis'), 
        fontsize=options.setdefault('ax_font_size',20))
    ax.set_ylabel(options.setdefault('ylabel','Y-axis'),
            fontsize=options.setdefault('ax_font_size',20))
    cbar = plt.colorbar(hst[3], ax=ax)
    if options['log_freq']:
        cbar.set_label(options.setdefault('clabel','Logarithmic counts'),
            fontsize=options['ax_font_size'])
    else:
        cbar.set_label(options.setdefault('clabel','Counts'),
            fontsize=options['ax_font_size'])

    plt.xticks(fontsize=options.setdefault('tick_font_size',16))
    plt.yticks(fontsize=options['tick_font_size'])
    if 'save' in options:
        fig.savefig(options['save'],bbox_inches='tight')
    else:
        if options.setdefault('show', False):
            plt.show()
        else:
            plt.draw()

    return fig, ax



def posterior(post, variables, **options):
    """This function creates several scatter plots of a set of orbital elements based on the
    different possible axis planar projections, calculates all possible permutations of plane
    intersections based on the number of columns

    :param numpy.ndarray post: Rows are distinct variable samples and columns are variables in the order of :code:`variables`
    :param list variables: Name of variables, used for axis names.
    :param options:  dictionary containing all the optional settings

    Currently the options fields are:
        :bins [int]: the number of bins
        :colormap [str]: Name of colormap to use.
        :title [string]: the title of the plot
        :title_font_size [int]: the title font size
        :axis_labels [list of strings]: labels for each column
        :tick_font_size [int]: the axis tick font size
        :window [tuple/list]: the size of the plot window in pixels (assuming dpi = 80)
        :save [string]: will not display figure and will instead save it to this path
        :show [bool]: if False will do draw() instead of show() allowing script to continue
        :tight_rect [list of 4 floats]: configuration for the tight_layout function


    """
    if type(post) != np.ndarray:
        post = np.array(post)

    #turn on TeX interperter
    plt.rc('text', usetex=True)

    lis = list(range(post.shape[1]))
    axis_plot = list(combinations(lis, 2))

    axis_label = variables
    
    if post.shape[1] == 2:
        subplot_cnt = (1,2)
        subplot_perms = 2
    elif post.shape[1] == 3:
        subplot_cnt = (1,3)
        subplot_perms = 3
    elif post.shape[1] == 4:
        subplot_cnt = (2,3)
        subplot_perms = 6
    elif post.shape[1] == 5:
        subplot_cnt = (2,5)
        subplot_perms = 10
    else:
        subplot_cnt = (3,5)
        subplot_perms = 15
    subplot_cnt_ind = 1

    if 'window' in options:
        size_in = options['window']
        size_in = tuple(x/80.0 for x in size_in)
    else:
        size_in=(19, 10)

    fig = plt.figure(figsize=size_in,dpi=80)

    cmap = options.setdefault('cmap', 'plasma')
    bins = options.setdefault('bins', int(np.round(np.sqrt(post.shape[0]))) )

    fig.suptitle(options.setdefault('title','Probability distribution'),
        fontsize=options.setdefault('title_font_size',24))
    axes = []
    for I in range( subplot_perms ):
        ax = fig.add_subplot(subplot_cnt[0],subplot_cnt[1],subplot_cnt_ind)
        axes.append(ax)
        x = post[:,axis_plot[I][0]]
        y = post[:,axis_plot[I][1]]
        fig, ax = hist2d(
            x.flatten(),
            y.flatten(),
            bins=bins,
            cmap=cmap,
            plot=(fig, ax),
            pdf=True,
            clabel='Probability',
            title = None,
            show=False,
        )
        ax.set_xlabel(axis_label[axis_plot[I][0]], 
            fontsize=options.setdefault('ax_font_size',22))
        ax.set_ylabel(axis_label[axis_plot[I][1]], 
            fontsize=options['ax_font_size'])
        plt.xticks(fontsize=options.setdefault('tick_font_size',17))
        plt.yticks(fontsize=options['tick_font_size'])
        subplot_cnt_ind += 1
    
    plt.tight_layout(rect=options.setdefault('tight_rect',[0, 0.03, 1, 0.95]))

    if 'save' in options:
        fig.savefig(options['save'],bbox_inches='tight')
    else:
        if options.setdefault('show', False):
            plt.show()
        else:
            plt.draw()

    return fig, axes



def hist(x, **options):
    """This function creates a histogram plot with lots of nice pre-configured settings unless they are overridden

    :param numpy.ndarray x:  data to histogram over, if x is not a vector it is flattened
    :param dict options: All keyword arguments as a dictionary containing all the optional settings.

    Currently the keyword arguments that can be used in the **options**:
        :bins [int]: the number of bins
        :density [bool]: convert counts to density in [0,1]
        :edges [float]: bin edge line width, set to 0 to remove edges
        :title [string]: the title of the plot
        :title_font_size [int]: the title font size
        :xlabel [string]: the label for the x axis
        :ylabel [string]: the label for the y axis
        :tick_font_size [int]: the axis tick font size
        :window [tuple/list]: the size of the plot window in pixels (assuming dpi = 80)
        :save [string]: will not display figure and will instead save it to this path
        :show [bool]: if False will do draw() instead of show() allowing script to continue
        :plot [tuple]: A tuple with the :code:`(fig, ax)` objects from matplotlib. Then no new figure and axis will be created.
        :logx [bool]: Determines if x-axis should be the logarithmic.
        :logy [bool]: Determines if y-axis should be the logarithmic.

    Example::
        
        import dpt_tools as dpt
        import numpy as np
        np.random.seed(19680221)

        x = 10*np.random.randn(1000)

        dpt.hist(x,
           title = "My first plot",
        )
        

    """

    if not isinstance(x, np.ndarray):
        _x = np.array(x)
    else:
        _x = x.copy()

    if _x.size != _x.shape[0]:
        _x = _x.flatten()

    if options.setdefault('logx', False ):
        _x = np.log10(_x)

    #turn on TeX interperter
    plt.rc('text', usetex=True)

    if 'window' in options:
        size_in = options['window']
        size_in = tuple(sz/80.0 for sz in size_in)
    else:
        size_in=(15, 7)
    if 'bin_size' in options:
        options['bins'] = int(np.round((np.max(_x)-np.min(_x))/options['bin_size']))

    if 'plot' in options:
        fig, ax = options['plot']
    else:
        fig, ax = plt.subplots(figsize=size_in,dpi=80)

    ax.hist(_x,
        options.setdefault('bins', int(np.round(np.sqrt(_x.size))) ),
        density=options.setdefault('density',False),
        facecolor=options.setdefault('color','b'),
        edgecolor='black',
        linewidth=options.setdefault('edges',1.2),
        cumulative=options.setdefault('cumulative',False),
        log=options.setdefault('logy', False ),
        label=options.setdefault('label', None ),
        alpha=options.setdefault('alpha',1.0),
    )
    ax.set_title(options.setdefault('title','Data scatter plot'),\
        fontsize=options.setdefault('title_font_size',24))
    ax.set_xlabel(options.setdefault('xlabel','X-axis'), \
        fontsize=options.setdefault('ax_font_size',20))
    if options['density']:
        ax.set_ylabel(options.setdefault('ylabel','Density'), \
            fontsize=options['ax_font_size'])
    else:
        ax.set_ylabel(options.setdefault('ylabel','Frequency'), \
            fontsize=options['ax_font_size'])
    plt.xticks(fontsize=options.setdefault('tick_font_size',16))
    plt.yticks(fontsize=options['tick_font_size'])
    if 'save' in options:
        fig.savefig(options['save'],bbox_inches='tight')
    else:
        if options.setdefault('show', False):
            plt.show()
        else:
            plt.draw()

    return fig, ax


def scatter(x, y, **options):
    """This function creates a scatter plot with lots of nice pre-configured settings unless they are overridden

    Currently the options fields are:
        :marker [char]: The marker type
        :size [int]: The size of the marker
        :alpha [float]: The transparency of the points.
        :title [string]: The title of the plot
        :title_font_size [int]: The title font size
        :xlabel [string]: The label for the x axis
        :ylabel [string]: The label for the y axis
        :tick_font_size [int]: The axis tick font size
        :window [tuple/list]: The size of the plot window in pixels (assuming dpi = 80)
        :save [string]: Will not display figure and will instead save it to this path
        :show [bool]: If False will do draw() instead of show() allowing script to continue
        :plot [tuple]: A tuple with the :code:`(fig, ax)` objects from matplotlib. Then no new figure and axis will be created.

    :param numpy.ndarray x:  x-axis data vector.
    :param numpy.ndarray y:  y-axis data vector.
    :param options:  dictionary containing all the optional settings.

    Example::
        
        import dpt_tools as dpt
        import numpy as np
        np.random.seed(19680221)

        x = 10*np.random.randn(100)
        y = 10*np.random.randn(100)

        dpt.scatter(x, y, {
            "title": "My first plot",
            } )

    """
    if type(x) != np.ndarray:
        x = np.array(x)
    if type(y) != np.ndarray:
        y = np.array(y)

    #turn on TeX interperter
    plt.rc('text', usetex=True)

    if 'window' in options:
        size_in = options['window']
        size_in = tuple(x/80.0 for x in size_in)
    else:
        size_in=(15, 7)

    if 'plot' in options:
        fig, ax = options['plot']
    else:
        fig, ax = plt.subplots(figsize=size_in,dpi=80)

    ax.scatter([x], [y],
        marker=options.setdefault('marker','.'),
        c=options.setdefault('color','b'),
        s=options.setdefault('size',2),
        alpha=options.setdefault('alpha',0.75),
    )
    ax.set_title(options.setdefault('title','Data scatter plot'),
        fontsize=options.setdefault('title_font_size',24))
    ax.set_xlabel(options.setdefault('xlabel','X-axis'),
        fontsize=options.setdefault('ax_font_size',20))
    ax.set_ylabel(options.setdefault('ylabel','Y-axis'),
        fontsize=options['ax_font_size'])
    plt.xticks(fontsize=options.setdefault('tick_font_size',16))
    plt.yticks(fontsize=options['tick_font_size'])
    if 'save' in options:
        fig.savefig(options['save'],bbox_inches='tight')
    else:
        if options.setdefault('show', False):
            plt.show()
        else:
            plt.draw()

    return fig, ax


def orbits(o, **options):
    """This function creates several scatter plots of a set of orbital elements based on the
    different possible axis planar projections, calculates all possible permutations of plane
    intersections based on the number of columns

    :param numpy.ndarray o: Rows are distinct orbits and columns are orbital elements in the order a, e, i, omega, Omega
    :param options:  dictionary containing all the optional settings

    Currently the options fields are:
        :marker [char]: the marker type
        :size [int]: the size of the marker
        :title [string]: the title of the plot
        :title_font_size [int]: the title font size
        :axis_labels [list of strings]: labels for each column
        :tick_font_size [int]: the axis tick font size
        :window [tuple/list]: the size of the plot window in pixels (assuming dpi = 80)
        :save [string]: will not display figure and will instead save it to this path
        :show [bool]: if False will do draw() instead of show() allowing script to continue
        :tight_rect [list of 4 floats]: configuration for the tight_layout function


    Example::
        
        import dpt_tools as dpt
        import numpy as np
        np.random.seed(19680221)

        orbs = np.matrix([
            11  + 3  *np.random.randn(1000),
            0.5 + 0.2*np.random.randn(1000),
            60  + 10 *np.random.randn(1000),
            120 + 5  *np.random.randn(1000),
            33  + 2  *np.random.randn(1000),
        ]).T

        dpt.orbits(orbs,
            title = "My orbital element distribution",
            size = 10,
        )


    """
    if type(o) != np.ndarray:
        o = np.array(o)

    if 'unit' in options:
        if options['unit'] == 'km':
            o[:,0] = o[:,0]/6353.0
        elif options['unit'] == 'm':
            o[:,0] = o[:,0]/6353.0e3
    else:
        o[:,0] = o[:,0]/6353.0

    #turn on TeX interperter
    plt.rc('text', usetex=True)

    lis = list(range(o.shape[1]))
    axis_plot = list(combinations(lis, 2))

    axis_label = options.setdefault('axis_labels', \
        [ "$a$ [$R_E$]","$e$ [1]","$i$ [deg]","$\omega$ [deg]","$\Omega$ [deg]","$M_0$ [deg]" ])
    
    if o.shape[1] == 2:
        subplot_cnt = (1,2)
        subplot_perms = 2
    elif o.shape[1] == 3:
        subplot_cnt = (1,3)
        subplot_perms = 3
    elif o.shape[1] == 4:
        subplot_cnt = (2,3)
        subplot_perms = 6
    elif o.shape[1] == 5:
        subplot_cnt = (2,5)
        subplot_perms = 10
    else:
        subplot_cnt = (3,5)
        subplot_perms = 15
    subplot_cnt_ind = 1

    if 'window' in options:
        size_in = options['window']
        size_in = tuple(x/80.0 for x in size_in)
    else:
        size_in=(19, 10)

    fig = plt.figure(figsize=size_in,dpi=80)

    fig.suptitle(options.setdefault('title','Orbital elements distribution'),\
        fontsize=options.setdefault('title_font_size',24))
    axes = []
    for I in range( subplot_perms ):
        ax = fig.add_subplot(subplot_cnt[0],subplot_cnt[1],subplot_cnt_ind)
        axes.append(ax)
        x = o[:,axis_plot[I][0]]
        y = o[:,axis_plot[I][1]]
        sc = ax.scatter( \
            x.flatten(), \
            y.flatten(), \
            marker=options.setdefault('marker','.'),\
            c=options.setdefault('color','b'),\
            s=options.setdefault('size',2))
        if isinstance(options['color'],np.ndarray):
            plt.colorbar(sc)
        x_ticks = np.linspace(np.min(o[:,axis_plot[I][0]]),np.max(o[:,axis_plot[I][0]]), num=4)
        plt.xticks( [round(x,1) for x in x_ticks] )
        ax.set_xlabel(axis_label[axis_plot[I][0]], \
            fontsize=options.setdefault('ax_font_size',22))
        ax.set_ylabel(axis_label[axis_plot[I][1]], \
            fontsize=options['ax_font_size'])
        plt.xticks(fontsize=options.setdefault('tick_font_size',17))
        plt.yticks(fontsize=options['tick_font_size'])
        subplot_cnt_ind += 1
    
    plt.tight_layout(rect=options.setdefault('tight_rect',[0, 0.03, 1, 0.95]))


    if 'unit' in options:
        if options['unit'] == 'km':
            o[:,0] = o[:,0]*6353.0
        elif options['unit'] == 'm':
            o[:,0] = o[:,0]*6353.0e3
    else:
        o[:,0] = o[:,0]*6353.0

    if 'save' in options:
        fig.savefig(options['save'],bbox_inches='tight')
    else:
        if options.setdefault('show', False):
            plt.show()
        else:
            plt.draw()

    return fig, axes

def _np_vec_norm(x,axis):
    return np.sqrt(np.sum(x**2,axis=axis))


e_lim = 1e-9
"""float: The limit on eccentricity below witch an orbit is considered circular
"""


i_lim = np.pi*1e-9
"""float: The limit on inclination below witch an orbit is considered not inclined.
"""


def cart2kep(x, m=0.0, M_cent=M_sol, radians=True):
    '''Converts set of Cartesian state vectors to set of Keplerian orbital elements.

    **Units:**
       All units are SI-units: `SI Units <https://www.nist.gov/pml/weights-and-measures/metric-si/si-units>`_

       Angles are by default given as radians, all angles are radians internally in functions, input and output angles can be both radians and degrees depending on the :code:`radians` boolean.

    **Orientation of the ellipse in the coordinate system:**
       * For zero inclination :math:`i`: the ellipse is located in the x-y plane.
       * The direction of motion as True anoamly :math:`\\nu`: increases for a zero inclination :math:`i`: orbit is anti-coockwise, i.e. from +x towards +y.
       * If the eccentricity :math:`e`: is increased, the periapsis will lie in +x direction.
       * If the inclination :math:`i`: is increased, the ellipse will rotate around the x-axis, so that +y is rotated toward +z.
       * An increase in Longitude of ascending node :math:`\Omega`: corresponds to a rotation around the z-axis so that +x is rotated toward +y.
       * Changing argument of perihelion :math:`\omega`: will not change the plane of the orbit, it will rotate the orbit in the plane.
       * The periapsis is shifted in the direction of motion.
       * True anomaly measures from the +x axis, i.e :math:`\\nu = 0` is located at periapsis and :math:`\\nu = \pi` at apoapsis.
       * All anomalies and orientation angles reach between 0 and :math:`2\pi`

       *Reference:* "Orbital Motion" by A.E. Roy.
    

    **Constants:**
       * :mod:`~dpt_tools.e_lim`: Used to determine circular orbits
       * :mod:`~dpt_tools.i_lim`: Used to determine non-inclined orbits


    **Variables:**
       * :math:`a`: Semi-major axis
       * :math:`e`: Eccentricity
       * :math:`i`: Inclination
       * :math:`\omega`: Argument of perihelion
       * :math:`\Omega`: Longitude of ascending node
       * :math:`\\nu`: True anoamly


    :param numpy.ndarray x: Cartesian state vectors where rows 1-6 correspond to :math:`x`, :math:`y`, :math:`z`, :math:`v_x`, :math:`v_y`, :math:`v_z` and columns correspond to different objects.
    :param float/numpy.ndarray m: Masses of objects. If m is a numpy vector of masses, the gravitational :math:`\mu` parameter will be calculated also as a vector.
    :param float M_cent: Is the mass of the central massive body, default value is the mass of the sun parameter in :mod:`~dpt_tools.M_sol`
    :param bool radians: If true radians are used, else all angles are given in degree.

    :return: Keplerian orbital elements where rows 1-6 correspond to :math:`a`, :math:`e`, :math:`i`, :math:`\omega`, :math:`\Omega`, :math:`\\nu` and columns correspond to different objects.
    :rtype: numpy.ndarray

    **Example:**
       
       Convert 1 AU distance object of Earth mass traveling at 30 km/s tangential velocity to Kepler elements.

       .. code-block:: python

          import dpt_tools as dpt
          import numpy as n
          import scipy.constants as c

          state = n.array([
            c.au*1.0, #x
            0, #y
            0, #z
            0, #vx
            30e3, #vy
            0, #vz
          ], dtype=n.float)

          orbit = dpt.cart2kep(state, m=5.97237e24, M_cent=1.9885e30, radians=False)
          print('Orbit: a={} AU, e={}'.format(orbit[0]/c.au, orbit[1]))
          print('(i, omega, Omega, nu)={} deg '.format(orbit[2:]))


    *Reference:* Daniel Kastinen Master Thesis: Meteors and Celestial Dynamics
    '''

    if not isinstance(x,np.ndarray):
        raise TypeError('Input type {} not supported: must be {}'.format( type(x),np.ndarray ))
    if x.shape[0] != 6:
        raise ValueError('Input data must have at least 6 variables along axis 0: input shape is {}'.format(x.shape))
    
    if len(x.shape) < 2:
        input_is_vector = True
        try:
            x.shape=(6,1)
        except ValueError as e:
            print('Error {} while trying to cast vector into single column.'.format(e))
            print('Input array shape: {}'.format(x.shape))
            raise
    else:
        input_is_vector = False

    ez = np.array([0,0,1], dtype=x.dtype)
    ex = np.array([1,0,0], dtype=x.dtype)
    
    o = np.empty(x.shape, dtype=x.dtype)
    iter_n = x.shape[1]

    r = x[:3,:]
    v = x[3:,:]
    rn = _np_vec_norm(r,axis=0)
    vn = _np_vec_norm(v,axis=0)

    mu = scipy.constants.G*(m + M_cent)

    vr = np.sum( (r/rn)*v , axis=0)

    epsilon = vn**2*0.5 - mu/rn

    o[0,:] = -mu/(2.0*epsilon)

    e = 1.0/mu*((vn**2 - mu/rn)*r - (rn*vr)*v)
    o[1,:] = _np_vec_norm(e,axis=0)

    #could implement this with nditers
    #np.nditer(a, flags=['external_loop'], order='F')

    for ind in range(iter_n):
        if o[0, ind] < 0:
            o[0, ind] = -o[0, ind]

        h = np.cross( r[:, ind], v[:, ind] )
        hn = np.linalg.norm(h)
        inc = np.arccos(h[2]/hn)

        if inc < i_lim:
            n = ex
            nn = 1.0
        else:
            n = np.cross(ez, h)
            nn = np.linalg.norm(n)

        if np.abs(inc) < i_lim:
            asc_node = 0.0
        elif n[1] < 0.0:
            asc_node = 2.0*np.pi - np.arccos(n[0]/nn)
        else:
            asc_node = np.arccos(n[0]/nn)

        if o[1,ind] < e_lim:
            omega = 0.0
        else:
            tmp_ratio = np.sum(n*e[:, ind], axis=0)/(nn*o[1, ind])
            if tmp_ratio > 1.0:
                tmp_ratio = 1.0
            elif tmp_ratio < -1.0:
                tmp_ratio = -1.0

            if e[2,ind] < 0.0:
                omega = 2.0*np.pi - np.arccos(tmp_ratio)
            else:
                omega = np.arccos(tmp_ratio)

        if o[1, ind] >= e_lim:
            tmp_ratio = np.sum(e[:,ind]*r[:,ind],axis=0)/(o[1, ind]*rn[ind])
            if tmp_ratio > 1.0:
                tmp_ratio = 1.0
            elif tmp_ratio < -1.0:
                tmp_ratio = -1.0

            if vr[ind] < 0.0:
                nu = 2.0*np.pi - np.arccos(tmp_ratio)
            else:
                nu = np.arccos(tmp_ratio)
        else:
            tmp_ratio = np.sum((n/nn)*(r[:,ind]/rn[ind]),axis=0)
            if tmp_ratio > 1.0:
                tmp_ratio = 1.0
            elif tmp_ratio < -1.0:
                tmp_ratio = -1.0
            nu = np.arccos(tmp_ratio)

            if (r[2,ind] < 0.0 and inc >= i_lim) or \
                    (r[1,ind] < 0.0 and inc < i_lim):
                nu = 2.0*np.pi - nu

        o[2,ind] = inc
        o[3,ind] = omega
        o[4,ind] = asc_node
        o[5,ind] = nu

    if not radians:
        o[2:,:] = np.degrees(o[2:,:])

    if input_is_vector:
        x.shape = (6,)
        o.shape = (6,)

    return o


def true2eccentric(nu, e, radians=True):
    '''Calculates the eccentric anomaly from the true anomaly.

    :param float/numpy.ndarray nu: True anomaly.
    :param float/numpy.ndarray e: Eccentricity of ellipse.
    :param bool radians: If true radians are used, else all angles are given in degrees

    :return: Eccentric anomaly.
    :rtype: numpy.ndarray or float
    '''
    if not radians:
        _nu = np.radians(nu)
    else:
        _nu = nu

    E = 2.0*np.arctan( np.sqrt( (1.0 - e)/(1.0 + e) )*np.tan(_nu*0.5) )
    E = np.mod(E + 2.*np.pi,2.*np.pi)
    
    if not radians:
        E = np.degrees(E)
    
    return E


def eccentric2true(E, e, radians=True):
    '''Calculates the true anomaly from the eccentric anomaly.

    :param float/numpy.ndarray E: Eccentric anomaly.
    :param float/numpy.ndarray e: Eccentricity of ellipse.
    :param bool radians: If true radians are used, else all angles are given in degrees

    :return: True anomaly.
    :rtype: numpy.ndarray or float
    '''
    if not radians:
        _E = np.radians(E)
    else:
        _E = E

    nu = 2.0*np.arctan( np.sqrt( (1.0 + e)/(1.0 - e) )*np.tan(_E*0.5) )
    nu = np.mod(nu + 2.*np.pi, 2.*np.pi)

    if not radians:
        nu = np.degrees(nu)
    
    return nu


def eccentric2mean(E,e,radians=True):
    '''Calculates the mean anomaly from the eccentric anomaly using Kepler equation.

    :param float/numpy.ndarray E: Eccentric anomaly.
    :param float/numpy.ndarray e: Eccentricity of ellipse.
    :param bool radians: If true radians are used, else all angles are given in degrees

    :return: Mean anomaly.
    :rtype: numpy.ndarray or float
    '''
    return E - e*np.sin(E)


def true2mean(nu, e, radians=True):
    '''Transforms true anomaly to mean anomaly.
    
    **Uses:**
       * :func:`~dpt_tools.true2eccentric`
       * :func:`~dpt_tools.eccentric2mean`

    :param float/numpy.ndarray nu: True anomaly.
    :param float/numpy.ndarray e: Eccentricity of ellipse.
    :param bool radians: If true radians are used, else all angles are given in degrees
    
    :return: Mean anomaly.
    :rtype: numpy.ndarray or float
    '''
    if not radians:
        _nu = np.radians(nu)
    else:
        _nu = nu

    E = true2eccentric(_nu, e, radians=True)
    M = eccentric2mean(E, e, radians=True)
    
    if not radians:
        M = np.degrees(M)

    return M


def elliptic_radius(E,a,e,radians=True):
    '''Calculates the distance between the left focus point of an ellipse and a point on the ellipse defined by the eccentric anomaly.

    :param float/numpy.ndarray E: Eccentric anomaly.
    :param float/numpy.ndarray a: Semi-major axis of ellipse.
    :param float/numpy.ndarray e: Eccentricity of ellipse.
    :param bool radians: If true radians are used, else all angles are given in degrees
    
    :return: Radius from left focus point.
    :rtype: numpy.ndarray or float
    '''
    if not radians:
        _E = np.radians(E)
    else:
        _E = E

    return a*(1.0 - e*np.cos( _E ))


def rot_mat_z(theta, dtype=np.float):
    '''Generates the 3D transformation matrix for rotation around Z-axis.
    
    :param float theta: Angle to rotate.
    :param numpy.dtype dtype: The data-type of the output matrix.

    :return: Rotation matrix
    :rtype: (3,3) numpy.ndarray
    '''
    R = np.zeros((3,3), dtype=dtype)
    R[0,0] = np.cos(theta)
    R[0,1] = -np.sin(theta)
    R[1,0] = np.sin(theta)
    R[1,1] = np.cos(theta)
    R[2,2] = 1.0
    return R


def rot_mat_x(theta, dtype=np.float):
    '''Generates the 3D transformation matrix for rotation around X-axis.
    
    :param float theta: Angle to rotate.
    :param numpy.dtype dtype: The data-type of the output matrix.

    :return: Rotation matrix
    :rtype: (3,3) numpy.ndarray
    '''
    R = np.zeros((3,3), dtype=dtype)
    R[1,1] = np.cos(theta)
    R[1,2] = -np.sin(theta)
    R[2,1] = np.sin(theta)
    R[2,2] = np.cos(theta)
    R[0,0] = 1.0
    return R


def rot_mat_y(theta, dtype=np.float):
    '''Generates the 3D transformation matrix for rotation around Y-axis.
    
    :param float theta: Angle to rotate.
    :param numpy.dtype dtype: The data-type of the output matrix.

    :return: Rotation matrix
    :rtype: (3,3) numpy.ndarray
    '''
    R = np.zeros((3,3), dtype=dtype)
    R[0,0] = np.cos(theta)
    R[0,2] = np.sin(theta)
    R[2,0] = -np.sin(theta)
    R[2,2] = np.cos(theta)
    R[1,1] = 1.0
    return R

# Leapseconds code from gdar by Norut/NORCE
# Contributed by Tom Grydeland <tgry@norceresearch.no>

sec = np.timedelta64(1000000000, 'ns')
'''numpy.datetime64: Interval of 1 second
'''


gps0_tai = np.datetime64('1980-01-06 00:00:19')
'''numpy.datetime64: Epoch of GPS time, in TAI
'''


leapseconds = np.array([
    '1972-01-01T00:00:00',
    '1972-07-01T00:00:00',
    '1973-01-01T00:00:00',
    '1974-01-01T00:00:00',
    '1975-01-01T00:00:00',
    '1976-01-01T00:00:00',
    '1977-01-01T00:00:00',
    '1978-01-01T00:00:00',
    '1979-01-01T00:00:00',
    '1980-01-01T00:00:00',
    '1981-07-01T00:00:00',
    '1982-07-01T00:00:00',
    '1983-07-01T00:00:00',
    '1985-07-01T00:00:00',
    '1988-01-01T00:00:00',
    '1990-01-01T00:00:00',
    '1991-01-01T00:00:00',
    '1992-07-01T00:00:00',
    '1993-07-01T00:00:00',
    '1994-07-01T00:00:00',
    '1996-01-01T00:00:00',
    '1997-07-01T00:00:00',
    '1999-01-01T00:00:00',
    '2006-01-01T00:00:00',
    '2009-01-01T00:00:00',
    '2012-07-01T00:00:00',
    '2015-07-01T00:00:00',
    '2017-01-01T00:00:00',
], dtype='M8[ns]')
'''numpy.ndarray: Leapseconds added since 1972. 

Must be maintained manually.

*Source:* `tai-utc <ftp://maia.usno.navy.mil/ser7/tai-utc.dat>`_
'''


def leapseconds_before(ytime, tai=False):
    '''Calculate the number of leapseconds has been added before given date.
    '''
    leaps = leapseconds_tai if tai else leapseconds

    if np.isscalar(ytime):
        return np.sum(leaps <= ytime) + 9
    else:
        rval = np.sum(leaps[np.newaxis, :] <= ytime.ravel()[..., np.newaxis], 1) + 9
        rval.shape = ytime.shape
        return rval


def tai2utc(ytime):
    '''TAI to UTC conversion using Leapseconds data.
    '''
    return ytime - leapseconds_before(ytime, tai=True)*sec

def utc2tai(ytime):
    '''UTC to TAI conversion using Leapseconds data.
    '''
    return ytime + leapseconds_before(ytime)*sec

# This can only be initialized after utc2tai has been defined.
leapseconds_tai = utc2tai(leapseconds)


def laguerre_solve_kepler(E0, M, e, tol=1e-12, degree=5):
    '''Solve the Kepler equation using the The Laguerre Algorithm, a algorithm that guarantees global convergence.
    Adjusted for solving only real roots (non-hyperbolic orbits)
    
    Absolute numerical tolerance is defined as :math:`|f(E)| < tol` where :math:`f(E) = M - E + e \sin(E)`.

    # TODO: implement hyperbolic solving.

    *Note:* Choice of polynomial degree does not matter significantly for convergence rate.

    :param float M: Initial guess for eccentric anomaly.
    :param float M: Mean anomaly.
    :param float e: Eccentricity of ellipse.
    :param float tol: Absolute numerical tolerance eccentric anomaly.
    :param int degree: Polynomial degree in derivation of Laguerre Algorithm.
    :return: Eccentric anomaly and number of iterations.
    :rtype: tuple of (float, int)

    *Reference:* Conway, B. A. (1986). An improved algorithm due to Laguerre for the solution of Kepler's equation. Celestial mechanics, 39(2), 199-211.

    **Example:**

    .. code-block:: python

       import dpt_tools as dpt
       M = 3.14
       e = 0.8
       
       #Use mean anomaly as initial guess
       E, iterations = dpt.laguerre_solve_kepler(
          E0 = M,
          M = M,
          e = e,
          tol = 1e-12,
       )
    '''

    degree = float(degree)

    _f = lambda E: M - E + e*np.sin(E)
    _fp = lambda E: e*np.cos(E) - 1.0
    _fpp = lambda E: -e*np.sin(E)

    E = E0

    f_eval = _f(E)

    it_num = 0

    while np.abs(f_eval) >= tol:
        it_num += 1
        #sqrt_term = np.sqrt(
        #    ((degree - 1.0)*fp(E))**2
        #    - degree*(degree - 1)*f(E)*fpp(E)
        #)

        fp_eval = _fp(E)

        sqrt_term = np.sqrt(np.abs(
            (degree - 1.0)**2*fp_eval**2
            - degree*(degree - 1.0)*f_eval*_fpp(E)
        ))

        denom_p = fp_eval + sqrt_term
        denom_m = fp_eval - sqrt_term

        if np.abs(denom_p) > np.abs(denom_m):
            delta = degree*f_eval/denom_p
        else:
            delta = degree*f_eval/denom_m

        E = E - delta

        f_eval = _f(E)

    return E, it_num


def _get_kepler_guess(M, e):
    '''The different initial guesses for solving the Kepler equation based on input mean anomaly
    
    :param float M: Mean anomaly.
    :param float e: Eccentricity of ellipse.
    :return: Guess for eccentric anomaly.
    :rtype: float

    Reference: Esmaelzadeh, R., & Ghadiri, H. (2014). Appropriate starter for solving the Kepler's equation. International Journal of Computer Applications, 89(7).
    '''
    if M > np.pi:
        _M = 2.0*np.pi - M
    else:
        _M = M

    if _M < 0.25:
        E0 = _M + e*np.sin(_M)/(1.0 - np.sin(_M + e) + np.sin(_M))
    elif _M < 2.0:
        E0 = _M + e
    else:
        E0 = _M + (e*(np.pi - _M))/(1.0 + e)

    if M > np.pi:
        E0 = 2.0*np.pi - E0

    return E0


def kepler_guess(M, e):
    '''Guess the initial iteration point for newtons method.
    
    :param float/numpy.ndarray M: Mean anomaly.
    :param float/numpy.ndarray e: Eccentricity of ellipse.
    :return: Guess for eccentric anomaly.
    :rtype: numpy.ndarray or float

    *Reference:* Esmaelzadeh, R., & Ghadiri, H. (2014). Appropriate starter for solving the Kepler's equation. International Journal of Computer Applications, 89(7).
    '''

    if isinstance(M, np.ndarray) or isinstance(e, np.ndarray):
        if not isinstance(M, np.ndarray):
            M = np.ones(e.shape, dtype=e.dtype)*M
        if not isinstance(e, np.ndarray):
            e = np.ones(M.shape, dtype=M.dtype)*e
        if M.shape != e.shape:
            raise TypeError('Input dimensions does not agree')

        E0 = np.empty(M.shape, dtype=M.dtype)

        out_it = E0.size
        Mit = np.nditer(M)
        eit = np.nditer(e)
        Eit = np.nditer(E0, op_flags=['readwrite'])

        for it in range(out_it):
            Mc = next(Mit)
            ec = next(eit)
            Ec = next(Eit)

            E_calc = _get_kepler_guess(Mc, ec)

            Ec[...] = E_calc

    else:
        E0 = _get_kepler_guess(M, e)

    return E0


def mean2eccentric(M, e, tol=1e-12, radians=True):
    '''Calculates the eccentric anomaly from the mean anomaly by solving the Kepler equation.

    :param float/numpy.ndarray M: Mean anomaly.
    :param float/numpy.ndarray e: Eccentricity of ellipse.
    :param float tol: Numerical tolerance for solving Keplers equation in units of radians.
    :param bool radians: If true radians are used, else all angles are given in degrees
    
    :return: True anomaly.
    :rtype: numpy.ndarray or float

    **Uses:**
       * :func:`~dpt_tools._get_kepler_guess`
       * :func:`~dpt_tools.laguerre_solve_kepler`
    '''

    if not radians:
        _M = np.radians(M)
    else:
        _M = M

    if isinstance(_M, np.ndarray) or isinstance(e, np.ndarray):
        if not isinstance(_M, np.ndarray):
            _M = np.ones(e.shape, dtype=e.dtype)*_M
        if not isinstance(e, np.ndarray):
            e = np.ones(_M.shape, dtype=_M.dtype)*e
        if _M.shape != e.shape:
            raise TypeError('Input dimensions does not agree')


        E = np.empty(_M.shape, dtype=_M.dtype)

        out_it = E.size
        Mit = np.nditer(_M)
        eit = np.nditer(e)
        Eit = np.nditer(E, op_flags=['readwrite'])

        for it in range(out_it):
            Mc = next(Mit)
            ec = next(eit)
            Ec = next(Eit)

            E0 = _get_kepler_guess(Mc, ec)
            E_calc, it_num = laguerre_solve_kepler(E0, Mc, ec, tol=tol)

            Ec[...] = E_calc

    else:
        if e == 0:
            return M

        E0 = _get_kepler_guess(_M, e)
        E, it_num = laguerre_solve_kepler(E0, _M, e, tol=tol)


    if not radians:
        E = np.degrees(E)

    return E


def mean2true(M, e, tol=1e-12, radians=True):
    '''Transforms mean anomaly to true anomaly.
    
    **Uses:**
       * :func:`~dpt_tools.mean2eccentric`
       * :func:`~dpt_tools.eccentric2true`

    :param float/numpy.ndarray M: Mean anomaly.
    :param float/numpy.ndarray e: Eccentricity of ellipse.
    :param float tol: Numerical tolerance for solving Keplers equation in units of radians.
    :param bool radians: If true radians are used, else all angles are given in degrees
    
    :return: True anomaly.
    :rtype: numpy.ndarray or float
    '''
    if not radians:
        _M = np.radians(M)
    else:
        _M = M

    E = mean2eccentric(_M, e, tol=tol, radians=True)
    nu = eccentric2true(E, e, radians=True)

    if not radians:
        nu = np.degrees(nu)

    return nu


def orbital_speed(r, a, mu):
    '''Calculates the orbital speed at a given radius for an Keplerian orbit :math:`v = \sqrt{\mu \left (\\frac{2}{r} - \\frac{1}{a} \\right )}`.
    
    :param float/numpy.ndarray r: Radius from the pericenter.
    :param float/numpy.ndarray a: Semi-major axis of ellipse.
    :param float mu: Standard gravitation parameter :math:`\mu = G(m_1 + m_2)` of the orbit.
    :return: Orbital speed.
    '''
    return np.sqrt(mu*(2.0/r - 1.0/a))


def orbital_period(a, mu):
    '''Calculates the orbital period of an Keplerian orbit :math:`v = 2\pi\sqrt{\\frac{a^3}{\mu}}`.
    
    :param float/numpy.ndarray a: Semi-major axis of ellipse.
    :param float mu: Standard gravitation parameter :math:`\mu = G(m_1 + m_2)` of the orbit.
    :return: Orbital period.
    '''
    return 2.0*np.pi*np.sqrt(a**3.0/mu)


def kep2cart(o, m=0.0, M_cent=M_sol, radians=True):
    '''Converts set of Keplerian orbital elements to set of Cartesian state vectors.
    
    **Units:**
       All units are SI-units: `SI Units <https://www.nist.gov/pml/weights-and-measures/metric-si/si-units>`_

       Angles are by default given as radians, all angles are radians internally in functions, input and output angles can be both radians and degrees depending on the :code:`radians` boolean.

       To use custom units, simply change the definition of :code:`mu = scipy.constants.G*(m + M_cent)` to an input parameter for the function as this is the only unit dependent calculation.

    **Orientation of the ellipse in the coordinate system:**
       * For zero inclination :math:`i`: the ellipse is located in the x-y plane.
       * The direction of motion as True anoamly :math:`\\nu`: increases for a zero inclination :math:`i`: orbit is anti-coockwise, i.e. from +x towards +y.
       * If the eccentricity :math:`e`: is increased, the periapsis will lie in +x direction.
       * If the inclination :math:`i`: is increased, the ellipse will rotate around the x-axis, so that +y is rotated toward +z.
       * An increase in Longitude of ascending node :math:`\Omega`: corresponds to a rotation around the z-axis so that +x is rotated toward +y.
       * Changing argument of perihelion :math:`\omega`: will not change the plane of the orbit, it will rotate the orbit in the plane.
       * The periapsis is shifted in the direction of motion.

       *Reference:* "Orbital Motion" by A.E. Roy.

    **Variables:**
       * :math:`a`: Semi-major axis
       * :math:`e`: Eccentricity
       * :math:`i`: Inclination
       * :math:`\omega`: Argument of perihelion
       * :math:`\Omega`: Longitude of ascending node
       * :math:`\\nu`: True anoamly

    **Uses:**
       * :func:`~dpt_tools.true2eccentric`
       * :func:`~dpt_tools.elliptic_radius`

    :param numpy.ndarray o: Keplerian orbital elements where rows 1-6 correspond to :math:`a`, :math:`e`, :math:`i`, :math:`\omega`, :math:`\Omega`, :math:`\\nu` and columns correspond to different objects.
    :param float/numpy.ndarray m: Masses of objects. If m is a numpy vector of masses, the gravitational :math:`\mu` parameter will be calculated also as a vector.
    :param float M_cent: Is the mass of the central massive body, default value is the mass of the sun parameter in :mod:`~dpt_tools.M_sol`
    :param bool radians: If true radians are used, else all angles are given in degree.

    :return: Cartesian state vectors where rows 1-6 correspond to :math:`x`, :math:`y`, :math:`z`, :math:`v_x`, :math:`v_y`, :math:`v_z` and columns correspond to different objects.
    :rtype: numpy.ndarray

    **Example:**
       
       Convert Earth J2000.0 orbital parameters to Cartesian position.

       .. code-block:: python

          import dpt_tools as dpt
          import numpy as n
          import scipy.constants as c

          #Periapsis approx 3 Jan
          orbit = n.array([
            c.au*1.000001018, #a
            0.0167086, #e
            7.155, #i
            288.1, #omega
            174.9, #Omega
            0.0, #nu
          ], dtype=n.float)

          state = dpt.kep2cart(orbit, m=5.97237e24, M_cent=1.9885e30, radians=False)
          print('Position: {} AU '.format(state[:3]/c.au))
          print('Velocity: {} km/s '.format(state[3:]/1e3))


    *Reference:* Daniel Kastinen Master Thesis: Meteors and Celestial Dynamics, "Orbital Motion" by A.E. Roy.
    '''
    if not isinstance(o,np.ndarray):
        raise TypeError('Input type {} not supported: must be {}'.format( type(o),np.ndarray ))
    if o.shape[0] != 6:
        raise ValueError('Input data must have at least 6 variables along axis 0: input shape is {}'.format(o.shape))
    
    if len(o.shape) < 2:
        input_is_vector = True
        try:
            o.shape=(6,1)
        except ValueError as e:
            print('Error {} while trying to cast vector into single column.'.format(e))
            print('Input array shape: {}'.format(o.shape))
            raise
    else:
        input_is_vector = False

    mu = scipy.constants.G*(m + M_cent)

    x = np.empty(o.shape, dtype=o.dtype)

    if not radians:
        o[2:,:] = np.radians(o[2:,:])

    nu = o[5,:]
    omega = o[3,:]
    asc_node = o[4,:]
    inc = o[2,:]
    wf = nu + omega
    e = o[1,:]
    a = o[0,:]

    Ecc = true2eccentric(nu,e,radians=True)

    rn = elliptic_radius(Ecc,a,e,radians=True)

    r = np.zeros( (3, o.shape[1]), dtype=o.dtype )
    r[0,:] = np.cos(wf)
    r[1,:] = np.sin(wf)
    r = rn*r

    cos_Omega = np.cos(asc_node)
    sin_Omega = np.sin(asc_node)
    cos_i = np.cos(inc)
    sin_i = np.sin(inc)
    cos_w = np.cos(omega)
    sin_w = np.sin(omega)

    #order is important not to change varaibles before they are used
    x_tmp = r[0,:].copy()
    r[2,:] = r[1,:]*sin_i
    r[0,:] = cos_Omega*r[0,:] - sin_Omega*r[1,:]*cos_i
    r[1,:] = sin_Omega*x_tmp  + cos_Omega*r[1,:]*cos_i

    l1 = cos_Omega*cos_w - sin_Omega*sin_w*cos_i
    l2 = -cos_Omega*sin_w - sin_Omega*cos_w*cos_i
    m1 = sin_Omega*cos_w + cos_Omega*sin_w*cos_i
    m2 = -sin_Omega*sin_w + cos_Omega*cos_w*cos_i
    n1 = sin_w*sin_i
    n2 = cos_w*sin_i
    b = a*np.sqrt(1.0 - e**2)
    n = np.sqrt(mu/a**3)
    nar = n*a/rn
    
    v = np.zeros( (3, o.shape[1]), dtype=o.dtype )
    bcos_E = b*np.cos(Ecc)
    asin_E = a*np.sin(Ecc)

    v[0,:] = l2*bcos_E - l1*asin_E
    v[1,:] = m2*bcos_E - m1*asin_E
    v[2,:] = n2*bcos_E - n1*asin_E

    v = nar*v

    x[:3,:] = r
    x[3:,:] = v

    if input_is_vector:
        x.shape = (6,)
        o.shape = (6,)
        if not radians:
            o[2:] = np.degrees(o[2:])
    else:
        if not radians:
            o[2:,:] = np.degrees(o[2:,:])


    return x


def gmst(mjd_UT1):
    '''Returns the Greenwich Mean Sidereal Time (rotation of the earth) at a specific UTC Modified Julian Date.
    Defined as the hour angle between the meridian of Greenwich and mean equinox of date at 0 h UT1
    
    :param float/numpy.ndarray mjd_UT1: UTC Modified Julian Date.
    :return: Greenwich Mean Sidereal Time in radians between 0 and :math:`2\pi`.
    
    *Reference:* Montenbruck & Gill: Satellite orbits
    '''
    frac = lambda a: a - np.floor(a)
    
    secs = 86400.0 # Seconds per day
    MJD_J2000 = 51544.5

    # Mean Sidereal Time
    mjd_0 = np.floor(mjd_UT1)
    UT1 = secs*(mjd_UT1 - mjd_0)
    T_0 = (mjd_0 - MJD_J2000)/36525.0
    T = (mjd_UT1 - MJD_J2000)/36525.0
    
    gmst = 24110.54841 \
        + 8640184.812866*T_0 \
        + 1.002737909350795*UT1 \
        + np.multiply(np.multiply((0.093104 - 6.2e-6*T), T), T)
                
    return 2*np.pi*frac(gmst/secs)

   
def date_to_jd(year,month,day):
    """Convert a date to Julian Day.
    
    :param int year: Year as integer. Years preceding 1 A.D. should be 0 or negative. The year before 1 A.D. is 0, 10 B.C. is year -9.
    :param int month: Month as integer, Jan = 1, Feb. = 2, etc.
    :param float day:  Day, may contain fractional part.

    :return: (float) Julian Day
    
    **Example:**

        Convert 6 a.m., February 17, 1985 to Julian Day
        
        .. code-block:: python

           >>> date_to_jd(1985,2,17.25)
           2446113.75
    

    *Reference:* 'Practical Astronomy with your Calculator or Spreadsheet', 4th ed., Duffet-Smith and Zwart, 2011.

    """
    if month == 1 or month == 2:
        yearp = year - 1
        monthp = month + 12
    else:
        yearp = year
        monthp = month
    
    # this checks where we are in relation to October 15, 1582, the beginning
    # of the Gregorian calendar.
    if ((year < 1582) or
        (year == 1582 and month < 10) or
        (year == 1582 and month == 10 and day < 15)):
        # before start of Gregorian calendar
        B = 0
    else:
        # after start of Gregorian calendar
        A = math.trunc(yearp / 100.)
        B = 2 - A + math.trunc(A / 4.)
        
    if yearp < 0:
        C = math.trunc((365.25 * yearp) - 0.75)
    else:
        C = math.trunc(365.25 * yearp)
        
    D = math.trunc(30.6001 * (monthp + 1))
    
    jd = B + C + D + day + 1720994.5
    
    return jd
    

def yearday_to_monthday(year_day, leap):
    """Convert a day of the year to a month-day pair.
    Only takes first order leap-year into account. The day of the year is actually counted so that it starts from 1, so that 1.0 corresponds to January 1 00:00.

    :param float year_day: Day of the year.
    :param bool leap: Indicates of the year is a leap year or not.

    :return: tuple of (month, day)
    
    **Example:**
        
        .. code-block:: python

           >>> yearday_to_monthday(1.1, False)
           (1.0, 1.1)
           >>> yearday_to_monthday(31.1, False)
           (1.0, 31.1)
           >>> yearday_to_monthday(32.1, False)
           (2.0, 1.1)


    """
    days_of_month = np.array([
        1,
        31,
        28,
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ], dtype=np.float)
    if leap:
        days_of_month[2] += 1
    cum_days = np.cumsum(days_of_month, dtype=np.float)
    month = np.argmax(year_day < cum_days) - 1
    day = year_day - (cum_days[month] - 1.0)
    return month + 1.0, day


def unix_to_jd(unix):
    '''Convert Unix time to JD UT1

    Constant is due to 0h Jan 1, 1970 = 2440587.5 JD

    :param float/numpy.ndarray unix: Unix time in seconds.
    :return: Julian Date UT1
    :rtype: float/numpy.ndarray
    '''
    return unix/86400.0 + 2440587.5


def npdt2date(dt):
    '''    
    Converts a numpy datetime64 value to a date tuple

    :param numpy.datetime64 dt: Date and time (UTC) in numpy datetime64 format

    :return: tuple (year, month, day, hours, minutes, seconds, microsecond)
             all except usec are integer
    '''

    t0 = np.datetime64('1970-01-01', 's')
    ts = (dt - t0)/sec

    it = int(np.floor(ts))
    usec = 1e6 * (dt - (t0 + it*sec))/sec

    tm = time.localtime(it)
    return tm.tm_year, tm.tm_mon, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec, usec


def npdt2mjd(dt):
    '''
    Converts a numpy datetime64 value (UTC) to a modified Julian date
    '''
    return (dt - np.datetime64('1858-11-17'))/np.timedelta64(1, 'D')


def mjd2npdt(mjd):
    '''
    Converts a modified Julian date to a numpy datetime64 value (UTC)
    '''
    day = np.timedelta64(24*3600*1000*1000, 'us')
    return np.datetime64('1858-11-17') + day * mjd


def unix2npdt(unix):
    return np.datetime64('1970-01-01') + np.timedelta64(1000*1000,'us')*unix

def npdt2unix(dt):
    return (dt - np.datetime64('1970-01-01'))/np.timedelta64(1,'s')


def jd_to_unix(jd_ut1):
    '''Convert JD UT1 time to Unix time
    
    Constant is due to 0h Jan 1, 1970 = 2440587.5 JD

    :param float/numpy.ndarray jd_ut1: Julian Date UT1
    :return: Unix time in seconds
    :rtype: float/numpy.ndarray
    '''
    return (jd_ut1 - 2440587.5)*86400.0


def jd_to_mjd(jd):
    '''Convert Julian Date (relative 12h Jan 1, 4713 BC) to Modified Julian Date (relative 0h Nov 17, 1858)
    '''
    return jd - 2400000.5


def mjd_to_jd(mjd):
    '''Convert Modified Julian Date (relative 0h Nov 17, 1858) to Julian Date (relative 12h Jan 1, 4713 BC)
    '''
    return mjd + 2400000.5


def mjd_to_j2000(mjd_tt):
    '''Convert from Modified Julian Date to days past J2000.
    
    :param float/numpy.ndarray mjd_tt: MJD in TT
    :return: Days past J2000
    :rtype: float/numpy.ndarray
    '''
    return mjd_to_jd(mjd_tt) - 2451545.0


def jd_to_date(jd):
    """Convert Julian Day to date.
    
    :param float jd: Julian Day
    :return: Tuple consisting of year, month and day
    :rtype: tuple

    **Return tuple:**
       :year: (int) Year as integer. Years preceding 1 A.D. should be 0 or negative. The year before 1 A.D. is 0, 10 B.C. is year -9.
       :month: (int) Month as integer, Jan = 1, Feb. = 2, etc.
       :day: (float) Day, may contain fractional part.

    **Example:**

       Convert Julian Day 2446113.75 to year, month, and day.

       .. code-block:: python

          >>> jd_to_date(2446113.75)
          (1985, 2, 17.25)
    

    *Reference:* 'Practical Astronomy with your Calculator or Spreadsheet', 4th ed., Duffet-Smith and Zwart, 2011.

    """
    jd = jd + 0.5
    
    F, I = math.modf(jd)
    I = int(I)
    
    A = math.trunc((I - 1867216.25)/36524.25)
    
    if I > 2299160:
        B = I + 1 + A - math.trunc(A / 4.)
    else:
        B = I
        
    C = B + 1524
    
    D = math.trunc((C - 122.1) / 365.25)
    
    E = math.trunc(365.25 * D)
    
    G = math.trunc((C - E) / 30.6001)
    
    day = C - E + F - math.trunc(30.6001 * G)
    
    if G < 13.5:
        month = G - 1
    else:
        month = G - 13
        
    if month > 2.5:
        year = D - 4716
    else:
        year = D - 4715
        
    return year, month, day


def _plot_orb_inst(res, orb_init, get_orbit):

    o = _gen_orbit(orb_init, res)
    x = np.empty(o.shape, dtype=o.dtype)
    for i in range(o.shape[1]):
        state = get_orbit(o[:,i])
        state.shape = (6,)
        x[:,i] = state

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(15, 5)

    max_range = orb_init[0]*1.1

    ax.plot([0,max_range],[0,0],[0,0],"-k")
    ax.plot([0,max_range],[0,0],[0,0],"-k", label='+x')
    ax.plot([0,0],[0,max_range],[0,0],"-b", label='+y')
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    ax.plot([0.0],[0.0],[0.0],"xk",alpha=1,label='Focus point')
    ax.plot(x[0,:],x[1,:],x[2,:],".k",alpha=0.5,label='Converted elements')
    ax.plot([x[0,0]],[x[1,0]],[x[2,0]],"or",alpha=1,label='$\\nu = 0$')
    ax.plot([x[0,int(res//4)]],[x[1,int(res//4)]],[x[2,int(res//4)]],"oy",alpha=1,label='$\\nu = 0.5\pi$')

    ax.legend()
    plt.title("Kep -> cart: a={} km, e={}, inc={} deg, omega={} deg, Omega={} deg ".format(
        orb_init[0]*1e-3,
        orb_init[1],
        orb_init[2],
        orb_init[3],
        orb_init[4],
    ))

def _gen_orbit(orb_init, res):
    o = np.empty((6,res),dtype=np.float)
    nu = np.linspace(0, 360.0, num=res, dtype=np.float)

    for i in range(res):
        o[:5,i] = orb_init
        o[5,i] = nu[i]
    return o


def ascending_node_from_statevector(sv, m, **kw):
    '''
    keywords include
      M_cent = 'central mass'
    '''
    cart = np.r_[sv['pos'], sv['vel']]
    a, e, inc, aop, raan, mu0p = cart2kep(cart, m=m, M_cent=M_earth, radians=False, **kw)
    mu0 = true2mean(mu0p, e, radians=False)

    ttan = find_ascending_node_time(a, e, aop, mu0, m, radians=False)
    return sec * ttan


def find_ascending_node_time(a, e, aop, mu0, m, radians=False):
    '''Find the time past the crossing of the ascending node.
    '''
    if radians:
        one_lap = np.pi*2.0
    else:
        one_lap = 360.0

    gravitational_param = scipy.constants.G*(M_earth + m)
    mean_motion = np.sqrt(gravitational_param/(a**3.0))/(np.pi*2.0)*one_lap

    true_at_asc_node = one_lap - aop
    mean_at_asc_node = true2mean(true_at_asc_node, e, radians=radians)

    md = mean_at_asc_node - mu0
    if md > 0:
        md -= one_lap

    time_from_asc_node = -md/mean_motion

    return time_from_asc_node


def plot_ref_orbit(get_orbit, res = 100, orb_init = np.array([10000e3, 0.2, 70, 120, 35], dtype=np.float)):
    '''Plots a specific reference orbit.

    :param function get_orbit: A function pointer that takes an Kepler state as input and returns a state vector.
    
    The arguments to :code:`get_orbit` should be:
      * *numpy.ndarray* of Kepler elements.
      * Use degrees for angles.
      * Kepler elements are :math:`a`, :math:`e`, :math:`i`, :math:`\omega`, :math:`\Omega`, :math:`\\nu`
      * See :func:`~dpt_tools.kep2cart`

    **Example:**

    .. code-block:: python

       import dpt_tools as dpt
       import space_object as so

       def get_orbit(o):
           obj=so.space_object(
               a=o[0],
               e=o[1],
               i=o[2],
               raan=o[4],
               aop=o[3],
               mu0=o[5],
               C_D=2.3,
               A=1.0,
               m=1.0
           )
           return obj.get_state([0.0])

       dpt.plot_ref_orbit(get_orbit)

    '''
    _plot_orb_inst(res, orb_init, get_orbit)
    plt.show()


def plot_orbit_convention(get_orbit, res = 100):
    '''Plots the orbit convention used by arbitrary function/program

    :param function get_orbit: A function pointer that takes an Kepler state as input and returns a state vector.
    
    The arguments to :code:`get_orbit` should be:
      * *numpy.ndarray* of Kepler elements.
      * Use degrees for angles.
      * Kepler elements are :math:`a`, :math:`e`, :math:`i`, :math:`\omega`, :math:`\Omega`, :math:`\\nu`
      * See :func:`~dpt_tools.kep2cart`

    **Example:**

    .. code-block:: python

       import dpt_tools as dpt
       import space_object as so

       def get_orbit(o):
           obj=so.space_object(
               a=o[0],
               e=o[1],
               i=o[2],
               raan=o[4],
               aop=o[3],
               mu0=o[5],
               C_D=2.3,
               A=1.0,
               m=1.0
           )
           return obj.get_state([0.0])

       dpt.plot_orbit_convention(get_orbit)

    '''
    a = 50000e3
    orb_init_list = []
    orb_init_list.append(np.array([a, 0, 0.0, 0.0, 0.0], dtype=np.float))
    orb_init_list.append(np.array([a, 0.8, 0.0, 0.0, 0.0], dtype=np.float))
    orb_init_list.append(np.array([a, 0.8, 45.0, 0.0, 0.0], dtype=np.float))
    orb_init_list.append(np.array([a, 0.8, 45.0, 90.0, 0.0], dtype=np.float))
    orb_init_list.append(np.array([a, 0.8, 45.0, 90.0, 90.0], dtype=np.float))

    for orb_init in orb_init_list:
        _plot_orb_inst(res, orb_init, get_orbit)

    plt.show()


def orbit3D(states, ax=None):
    '''Create a 3D plot of a set of states.
    '''
    if ax is None:
        fig = plt.figure(figsize=(15,15))
        ax = fig.add_subplot(111, projection='3d')

    plothelp.draw_earth_grid(ax)
    ax.plot(states[0,:], states[1,:], states[2,:],"-b", alpha=0.5)
    max_range = np.max(np.abs(states.flatten()))*1.1
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    
    return ax
