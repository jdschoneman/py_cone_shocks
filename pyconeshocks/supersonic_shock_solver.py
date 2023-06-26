
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


GAMMA = 1.4


def post_oblique_assumed(M, shock_angle, gamma = GAMMA):
    """
    Obtain downstream conditions for oblique shock of assumed angle given the
    upstream Mach number. Returns the downstream Mach number, deflection
    angle, and ratio of total pressures across the shock.

    Equations taken from:

        https://www.grc.nasa.gov/www/k-12/airplane/oblique.html

    Parameters
    ----------
    M : float
        Upstream Mach number.
    shock_angle : float
        DESCRIPTION.
    gamma : float, optional
        Ratio of specific heats. The default is GAMMA.

    Returns
    -------
    M1 : float
        Downstream Mach number.
    a : float
        Flow deflection angle across shock
    p0_ratio : float
        Ratio of total pressures across shock; P0_downstream/P0_upstream

    """

    # Deflection angle
    cot_a = np.tan(shock_angle) * ((gamma + 1) * M**2 / (2*(M**2*np.sin(shock_angle)**2 - 1)) - 1)
    a = np.arctan(1./cot_a)

    # Downstream Mach
    M1_rhs = ((gamma - 1.)*M**2 *np.sin(shock_angle)**2 + 2) / (2 * gamma * M**2 * np.sin(shock_angle)**2 - (gamma - 1))
    M1 = np.sqrt(M1_rhs/np.sin(shock_angle - a)**2)

    # Total pressure ratio
    p0_A = (gamma + 1)*(M*np.sin(shock_angle))**2/((gamma - 1)*(M*np.sin(shock_angle))**2 + 2)
    p0_B = (gamma + 1)/(2*gamma*(M*np.sin(shock_angle))**2 - (gamma - 1))
    p0_ratio = p0_A**(gamma/(gamma - 1))*p0_B**(1/(gamma - 1))

    return M1, a, p0_ratio


## Cone shock functions
def tm_fun(theta, y, gamma = GAMMA):
    """
    Taylor-Maccoll function
    Source: https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/shockc/

    This is identical to Eq. (13.80) of Anderson, HOWEVER Anderson makes clear
    that the eq. is written in terms of a nondimensionalized velocity
    V/Vmax = V' = 1/sqrt(2/((gamma - 1)*M^2) + 1)

    where Vmax is the velocity associated with expansion to vacuum. This
    affects the input initial condition for the ODE.

    We are solving the ordinary differential equation:

    dVr/dtheta = Vr' = Vtheta
    dVtheta/dtheta = d^2Vr/dtheta^2 = Vr'' = f(Vr, Vtheta)

    The function for the 2nd derivative can be written as:
    A*B*(c1 + c2 + Vr'') = d1 + d2*Vr''
    Vr'' = (d1 - A*B*(c1 + c2))/(A*B - d2)

    A = (gamma - 1)/2
    B = 1 - Vr^2 - (V')^2
    c1 = 2*Vr
    c2 = cot(theta)*V'
    d1 = Vr'^2*Vr
    d2 = Vr'^2

    The states are [Vr, Vtheta = Vr'']

    Parameters
    ----------
    theta : float
        Wave angle from cone centerline [rad]
    y : ndarray
        ODE state vector [Vr, Vtheta]
    gamma : float, optional
        Ratio of specific heats. The default is GAMMA.

    Returns
    -------
    dy : list
        Derivatives of the state vector

    """

    Vr, Vtheta = y
    dVr = Vtheta

    A = (gamma - 1.)*0.5
    B = 1 - Vr*Vr - dVr*dVr
    c1 = 2*Vr
    c2 = dVr/np.tan(theta)
    d1 = dVr*dVr*Vr
    d2 = dVr*dVr

    d2Vr = (d1 - A*B*(c1 + c2))/(A*B - d2)

    return [dVr, d2Vr]


def event_fun(theta, y):
    """
    Monitors for the condition Vtheta = 0 which indicates the cone "surface"
    has been reached. Should be set as terminal to terminate the ODE
    integration.

    Parameters
    ----------
    theta : float
        Wave angle from cone centerline [rad]
    y : ndarray
        ODE state vector [Vr, Vtheta]

    Returns
    -------
    event : float
        This is just Vtheta; integration terminates when Vtheta crosses zero.

    """
    return y[1]

event_fun.terminal = True

def solve_cone_shock(M, shock_angle, gamma = GAMMA, n = 100,
                     angle_only = False):
    """
    Solves the conical shock field for given upstream Mach number and shock
    angle. Returns the vector of angles from surface to shockwave along
    with the nondimensional velocity and Mach number distribution between the
    shock and the surface.

    For 0 = upstream and 1 = downstream conditions
    p00/p0 = (1 + 0.5*(gamma - 1)*M0**2)**(gamma/(gamma - 1))
    p01/p00 = p0r
    p01/p1(theta) = (1 + 0.5*(gamma - 1)*M1(theta)**2)**(gamma/(gamma - 1))

    p01/p00 = p0r = (1 + 0.5*(gamma - 1)*M1(theta)**2)**(gamma/(gamma - 1))*p1(theta)/(1 + 0.5*(gamma - 1)*M0**2)**(gamma/(gamma - 1)) * p0
    p1(theta)/p0 = p0r*(1 + 0.5*(gamma - 1)*M0**2)**(gamma/(gamma - 1))/(1 + 0.5*(gamma - 1)*M1(theta)**2)**(gamma/(gamma - 1))



    Parameters
    ----------
    M : float
        Upstream Mach number
    shock_angle : float
        Assumed shock angle [rad]
    gamma : float, optional
        Specific heat ratio. The default is GAMMA.
    n : int, optional
        Number of partitions to use in the returned solution from surface_angle
        to shock_angle. The default is 100.
    angle_only : bool, optional
        Flag to compute the surface angle only, for use in zeroing against
        a fixed cone angle.

    Returns
    -------
    theta_surface : float
        Surface angle value where Vtheta = 0. If angle_only is True, this
        is the only returned value.
    theta_vector : ndarray
        Vector of length n from surface_angle to shock_angle. Only returned if
        angle_only is False.
    mach_vector : ndarray
        Vector of length n giving Mach variation from surface to shock. Only
        returned if angle_only is False.
    v_vector : ndarray
        Vector of size (n, 2) giving nondimensional (Vr, Vtheta) variation from
        surface to shock. Only returned if angle_only is False.
    p_vector : ndarray
        Vector of size (n, 2) giving the static pressure ratio p/pinf with
        respect to freestream Mach number. Only returned if angle_only is
        False.
    """

    # Use oblique shock relations for post-
    # _, p2, T2 = postOblique(shock_angle, cone_angle, Ma, p, T, 1.4)
    M2, a, p0r = post_oblique_assumed(M, shock_angle)
    if np.isnan(M2) or M2 < 0: raise ValueError('Invalid post-shock M2')

    # Normalized ratio V/Vmax
    Vp = 1/np.sqrt(2/((gamma - 1)*M2**2) + 1)

    Vx = Vp*np.cos(a)
    Vy = Vp*np.sin(a)
    Vtheta = Vy*np.cos(shock_angle) - Vx*np.sin(shock_angle)
    Vr = Vy*np.sin(shock_angle) + Vx*np.cos(shock_angle)
    y0 = [Vr, Vtheta]

    sol = solve_ivp(tm_fun, [shock_angle, 0.], y0, dense_output=True,
                    events = [event_fun])

    if angle_only is True:
        return sol.t_events[0]

    # Flowfield calculations
    theta_vector = np.linspace(sol.t[-1], sol.t[0], n)
    v_vector = sol.sol(theta_vector).T
    Vp = np.linalg.norm(v_vector, axis = 1)

    mach_vector = np.sqrt(2*Vp**2/(1 - Vp**2)/(gamma - 1))

    p_vector = p0r*(1 + 0.5*(gamma - 1)*M**2)**(gamma/(gamma - 1))/((1 + 0.5*(gamma - 1)*mach_vector**2)**(gamma/(gamma - 1)))

    return theta_vector, mach_vector, v_vector, p_vector

def solve_cone_flow(M, cone_angle, gamma = GAMMA, n = 100):
    """
    Obtain shock angle and flowfield results for a given upstream Mach number
    and cone semi-vertex angle.

    Parameters
    ----------
    M : float
        Upstream Mach number
    cone_angle : float
        Cone semi-vertex angle [rad]
    gamma : float, optional
        Specific heat ratio. The default is GAMMA.
    n : int, optional
        Number of partitions to use in the returned solution from surface_angle
        to shock_angle. The default is 100.

    Returns
    -------
    theta_vector : ndarray
        Vector of length n from cone_angle to shock_angle.
    mach_vector : ndarray
        Vector of length n giving Mach variation from surface to shock.
    v_vector : ndarray
        Vector of size (n, 2) giving nondimensional (Vr, Vtheta) variation from
        surface to shock.
    p_vector : ndarray
        Vector of size (n, 2) giving the static pressure ratio p/pinf with
        respect to freestream Mach number.

    """

    # Set up routine to zero shock solution on desired cone angle
    angle_fun = lambda shock_angle: (solve_cone_shock(M, shock_angle[0], gamma = gamma,
                                                      angle_only = True)
                                     - cone_angle)

    # Call solver, using slight increase on Mach angle as initial guess
    mach_angle = np.arcsin(1/M)
    shock_angle = fsolve(angle_fun, mach_angle*1.0001)[0]

    # Call again with known shock angle and get surface outputs
    return solve_cone_shock(M, shock_angle, gamma = gamma, n = n)

def pratio_to_cp(mach, pratio, gamma = GAMMA):
    """

    Use the static pressure ratio p/pinf to generate a pressure coefficient;

    cp = (p - pinf)/qinf
       = (p/pinf - 1)/(qinf/pinf)
       = (p/pinf - 1)/(0.5*gamma*mach**2)

    Parameters
    ----------
    mach : float
        Freestream Mach number
    pratio : ndarray or float
        Static pressure ratio p/pinf
    gamma : float, optional
        Specific heat ratio. The default is GAMMA

    Returns
    -------
    cp : ndarray
        Vector of pressure coefficient values
    """

    return (pratio - 1.)/(0.5*gamma*mach**2)

# TODO: This could be generated during the solution integration, although
# would have to be in the spherical coordinate system so a change of
# coords would be needed
def get_characteristic(mach, theta_vector, v_vector, x_start = 1.0,
                       dx_char = 0.01):
    """
    Generate the trailing left-running characteristic line for the conical
    flow. The parameter x_start specifies the length of the cone, and the
    characteristic line is generated from this point until it intersects with
    the conical shock-wave. The computed characteristic is the left-running
    line given by Eq. (13.10) of Anderson 5e:

        (dy/dx)_char = tan(beta + mu)

    Where beta = tan(Vy/Vx) is the STREAMLINE angle (replacing theta in the
    textbook, since theta is used here as the spherical coordinate system for
    the cone) and mu = arcsin(1/M)

    Parameters
    ----------
    mach : float
        Upstream Mach number
    theta_vector : ndarray
        Theta solution vector from the cone surface angle to the shock surface
        angle
    v_vector : ndarray
        Array of (vr, vtheta) normalized velocities; same length as theta_vector
    x_start : float, optional
        Starting x-location of the cone base. The default is 1.0.

    Returns
    -------
    char_x : ndarray
        X coordinates of the characteristic line
    char_y : ndarray
        Y coordinates of the characteristic line
    """

    # Some convenience values
    theta_cone = theta_vector[0]
    theta_shock = theta_vector[-1]

    # Solution vectors appended to during each
    char_x = [x_start, ]
    char_y = [x_start*np.tan(theta_cone), ]
    theta = theta_cone
    while theta < theta_shock:

        vr = np.interp(theta, theta_vector, v_vector[:, 0])
        vtheta = np.interp(theta, theta_vector, v_vector[:, 1])

        vx = vr*np.cos(theta) - vtheta*np.sin(theta)
        vy = vr*np.sin(theta) + vtheta*np.cos(theta)

        beta = np.tan(vy/vx)
        mu = np.arcsin(1/M)
        dy = np.tan(beta + mu)*dx_char

        char_x.append(char_x[-1] + dx_char)
        char_y.append(char_y[-1] + dy)
        theta = np.tan(char_y[-1]/char_x[-1])

    return np.array(char_x), np.array(char_y)


def plot_shock_solution(mach, cone_angle, value = 'cp',  ax = None,
                        nr = 101, nt = 100, r_shock_max = 1.5):
    """
    Generates a cone shock plot based on the input Mach number and cone angle.


    Parameters
    ----------
    mach : float
        Upstream Mach number
    cone_angle : float
        Cone angle
    value : str
        Solution value to plot; default is 'cp' for pressure coefficient.
    ax : PolarAxesSubplot, optional
        Axis for subplotting. The default is None, in which case a new figure
        and axis is generated.

    Returns
    -------
    ax : PolarAxes Subplot
        Plot axis; redundant return if this was supplied as an input argument
    theta_vector : ndarray
        Vector of length n from cone_angle to shock_angle.
    mach_vector : ndarray
        Vector of length n giving Mach variation from surface to shock.
    v_vector : ndarray
        Vector of size (n, 2) giving nondimensional (Vr, Vtheta) variation from
        surface to shock.
    p_vector : ndarray
        Vector of size (n, 2) giving the static pressure ratio p/pinf with
        respect to freestream Mach number.
    """

    # Obtain solution
    theta_vector, mach_vector, v_vector, p_vector = solve_cone_flow(M, cone_angle,
                                                                        n = nt)

    # Generate plotting geometry arrays
    r_cone = np.linspace(0., 1., nr)
    r_shock = np.linspace(0., r_shock_max, nr)
    th_cone = np.linspace(0., np.pi/2, nt)
    R, TH = np.meshgrid(r_shock, th_cone)
    A, B = np.mgrid[:nr, :nt]

    # Project solution into polar space
    if value == 'cp':
        sol_cone = np.interp(th_cone, theta_vector, pratio_to_cp(M, p_vector),
                            left = np.nan, right = 0.0)
    else:
        raise NotImplementedError('No implementation for value %s' % value)

    SOL = sol_cone[B]

    # Axis check
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize = (10, 8),
                               subplot_kw={'projection': 'polar'},
                               constrained_layout = True)

    # Plot cone, shock, & characteristic line
    ax.plot(theta_vector[0]*np.ones(nr, ), r_cone, 'k-')
    ax.plot(theta_vector[-1]*np.ones(nr, ), r_shock, 'b-')
    x_base = r_cone[-1]*np.cos(theta_vector[0])*np.ones(nr, )
    y_base = np.linspace(0., r_cone[-1]*np.sin(theta_vector[0]), nr)
    ax.plot(np.tan(y_base/x_base), np.sqrt(y_base**2 + x_base**2), 'k-')
    char_x, char_y = get_characteristic(M, theta_vector, v_vector,
                                        x_start = x_base[0])
    ax.plot(np.tan(char_y/char_x), np.sqrt(char_y**2 + char_x**2), 'r--')

    # Plot contours
    g = ax.contourf(th_cone, r_shock, SOL, levels = 10)
    ax.contour(th_cone, r_shock, SOL, levels = 10, colors = 'k',
               linewidths = 0.5)

    # Axis decorations
    plt.colorbar(mappable = g, ax = ax)
    ax.set_xlim(0., np.pi/2)
    ax.set_ylim(0., r_shock[-1])
    ax.set_title('Shock Solution for M%.2f; Cone Angle %.2f deg'
                 % (mach, cone_angle/np.pi*180.),
                 fontweight = 'bold')

    return ax, theta_vector, mach_vector, v_vector, p_vector


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    # For a fixed cone angle, plot flow velocity vectors for a few Mach numbers
    cone_angle = 5./180*np.pi
    # machs = [1.5, 3.0, 5.0]
    machs = [3., 5., ]
    gamma = GAMMA

    fig, axes = plt.subplots(1, len(machs), figsize = (5*len(machs) + 4, 8),
                             subplot_kw={'projection': 'polar'},
                             constrained_layout = True)
    if not hasattr(axes, '__len__'):
        axes = [axes, ]


    for M, ax in zip(machs, axes):

        _, theta_vector, mach_vector, v_vector, p_vector = plot_shock_solution(M,
                                                                               cone_angle,
                                                                               ax = ax)


        # Debug -- points from Sims 1964
        if M == 3.:
            scale = np.sin(theta_vector[0])/0.5
            x_tab = scale*np.array([5.715, 5.9672722, 6.2557012, 6.8018027, 7.6016621, 8.0569446])
            r_tab = scale*np.array([0.5, 0.61665729, 0.74594683, 0.98318391, 1.3198670, 1.5078154])
            ax.plot(np.tan(r_tab/x_tab), np.sqrt(x_tab**2 + r_tab**2),
                    'ko')

        if M == 5.:
            scale = np.sin(theta_vector[0])/0.5
            x_tab = scale*np.array([5.715, 6.0869315, 7.4260095])
            r_tab = scale*np.array([0.5, .61292148, 0.99413879])
            ax.plot(np.tan(r_tab/x_tab), np.sqrt(x_tab**2 + r_tab**2),
                    'ko')

