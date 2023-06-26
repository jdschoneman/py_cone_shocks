# -*- coding: utf-8 -*-
"""

Replicate Charts 5, 6, and 7 of NACA Report 1135:

    https://www.grc.nasa.gov/www/k-12/airplane/Images/naca1135.pdf

These provide the variation of shockwave angle, surface pressure coefficient,
and surface Mach number for various upstream Mach numbers as functions of
the cone semi-vertex angle.


"""
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


from supersonic_shock_solver import solve_cone_shock, pratio_to_cp, GAMMA

mvec = [1.05, 1.1, 1.15, 1.2, 1.25, 1.35, 1.4, 1.45, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
        2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.5, 5., 6., 8., 10.,
        20, 1000.]
mach_labels = [1.05, 1.1, 1.15, 1.2, 1.25, 1.4, 1.5, 1.7,
               2.0, 2.4, 3.0, 5.0, 20.0]
ntries = 250
max_angle = np.pi/2

# Figure 5 - shockwave angle
fig_angle, ax_angle = plt.subplots(1, 1, figsize = (14, 8),
                                   constrained_layout = True)

# Figure 6 - surface Cp
fig_cp, ax_cp = plt.subplots(1, 1, figsize = (14, 8),
                                    constrained_layout = True)

# Figure 7 - surface Mach number
fig_surfmach, ax_surfmach = plt.subplots(1, 1, figsize = (14, 8),
                                          constrained_layout = True)


max_angles = [max_angle, ]
max_surf_angles = [0., ]
min_surf_mach = [1., ]
max_surf_press = [0., ]
LW = 1.0

for mach in mvec:
    mach_angle = np.arcsin(1/mach)
    shock_angle_attempts = np.linspace(mach_angle*1.00001, np.pi/2, ntries)
    shock_angles = [mach_angle, ]
    surface_angles = [0., ]
    surface_mach = [mach, ]
    surface_pratio = [1., ]
    print('Mach = %f' % mach)
    for shock_angle in shock_angle_attempts:
        # try:
        flow_angles, mach_vector, _, p_vector = solve_cone_shock(mach, shock_angle, n = 3)
        if mach_vector[-1] <= 1.:
            break
        shock_angles.append(shock_angle)
        surface_angles.append(flow_angles[0])
        surface_mach.append(mach_vector[0])
        surface_pratio.append(p_vector[0])
        # except ValueError:
        #     print('\tFailed for shock angle %.2f deg' % (shock_angle/np.pi*180))
        #     continue

    max_angle = shock_angles[-1]
    max_angles.append(max_angle)
    max_surf_angles.append(surface_angles[-1])
    min_surf_mach.append(surface_mach[-1])
    max_surf_press.append(surface_pratio[-1])

    ax_angle.plot(np.array(surface_angles)*180/np.pi,
                  np.array(shock_angles)*180/np.pi, 'k-',
                  linewidth = LW)


    cp = pratio_to_cp(mach, np.array(surface_pratio))
    ax_cp.plot(np.array(surface_angles)*180/np.pi,
               cp, 'k-', linewidth = LW)

    mach_param = 1. - 1./np.array(surface_mach)
    ax_surfmach.plot(np.array(surface_angles)*180/np.pi,
                     mach_param, 'k-', linewidth = LW)


    if mach in mach_labels:
        ax_angle.text(max_surf_angles[-1]/np.pi*180, max_angles[-1]/np.pi*180,
                      '%.2f' % mach, va = 'bottom')
        ax_cp.text(max_surf_angles[-1]/np.pi*180, (max_surf_press[-1] - 1)/(0.5*GAMMA*mach**2),
                      '%.2f' % mach, va = 'bottom', ha = 'right')
        ax_surfmach.text(max_surf_angles[-1]/np.pi*180, 1 - 1/min_surf_mach[-1],
                      '%.2f' % mach, va = 'top', ha = 'right')

ax_angle.plot(np.array(max_surf_angles)/np.pi*180,
              np.array(max_angles)/np.pi*180, 'k--')


ax_cp.plot(np.array(max_surf_angles[1:])/np.pi*180,
           pratio_to_cp(np.array(mvec), np.array(max_surf_press[1:])),
           'k--')

ax_surfmach.plot(np.array(max_surf_angles)/np.pi*180,
                 1 - 1/np.array(min_surf_mach), 'k--')


# ax_angle.grid(which = 'major')
for ax in [ax_angle, ax_cp, ax_surfmach]:
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, which = 'both')
    ax.set_xlim([0., 58.])
    ax.set_xlabel('Cone Semi-Vertex Angle [deg]', fontweight = 'bold')

ax_angle.set_ylim([0., 90.])
ax_angle.set_ylabel('Shockwave Angle [deg]', fontweight = 'bold')
ax_angle.set_title('Variation of Shock-Wave Angle with Cone Semi-Vertex Angle for Various Upstream Mach Numbers; $\mathbf{\gamma = %.1f}$' % GAMMA
                   + '\n(Replication of NACA 1135 Chart 5)', fontweight = 'bold')
fig_angle.savefig('../images/chart5.png', bbox_inches = 'tight')

ax_cp.set_ylim([0., 2.])
ax_cp.set_ylabel('Surface Pressure Coefficient $\mathbf{(p_c - p_{\infty})/q_{\infty}}$', fontweight = 'bold')
ax_cp.set_title('Variation of Surface Pressure Coefficient with Cone Semi-Vertex Angle for Various Upstream Mach Numbers; $\mathbf{\gamma = %.1f}$' % GAMMA
                   + '\n(Replication of NACA 1135 Chart 6)', fontweight = 'bold')
fig_cp.savefig('../images/chart6.png', bbox_inches = 'tight')

ax_surfmach.set_ylim([-1., 1.])
ax_surfmach.set_ylabel('Surface Mach Parameter $\mathbf{1 - 1/M_c}$', fontweight = 'bold')
ax_surfmach.set_title('Variation of Surface Mach Number with Cone Semi-Vertex Angle for Various Upstream Mach Numbers; $\mathbf{\gamma = %.1f}$' % GAMMA
                   + '\n(Replication of NACA 1135 Chart 7)', fontweight = 'bold')
fig_surfmach.savefig('../images/chart7.png', bbox_inches = 'tight')


