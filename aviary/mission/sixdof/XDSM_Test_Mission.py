"""
XDSM diagram for VTOL trajectory optimization using pyXDSM
Represents the multiphase 6-DOF trajectory optimization problem
"""

from pyxdsm.XDSM import XDSM, OPT, SOLVER, FUNC, LEFT

# Create XDSM object
x = XDSM()

# Add optimizer (SNOPT)
x.add_system("opt", OPT, r"\text{SNOPT}")

# Add trajectory components for each phase
x.add_system("climb", SOLVER, r"\text{Climb Phase}")
x.add_system("cruise", SOLVER, r"\text{Cruise Phase}")
x.add_system("descent", SOLVER, r"\text{Descent Phase}")

# Add ODE components (shown for one phase, but applies to all)
x.add_system("ode", FUNC, r"\text{vtolODE}")
x.add_system("atmos", FUNC, r"\text{Atmosphere}")
x.add_system("aero", FUNC, r"\text{Aerodynamics}")
x.add_system("forces", FUNC, r"\text{Force Resolver}")
x.add_system("eom", FUNC, r"\text{6-DOF EOM}")

# Optimizer to phases - design variables
x.connect("opt", "climb", r"$T_x, T_y, T_z, l_x, l_y, l_z$")
x.connect("opt", "cruise", r"$T_x, T_y, T_z, l_x, l_y, l_z$")
x.connect("opt", "descent", r"$T_x, T_y, T_z, l_x, l_y, l_z$")

# Phase to ODE - states and controls
x.connect("climb", "ode", [r"$u, v, w$", r"$\phi, \theta, \psi$", r"$x, y, z$", r"$T_x, T_y, T_z$"])

# ODE internal data flow
x.connect("ode", "atmos", r"$h = -z$")
x.connect("atmos", "aero", r"$\rho, sos$")
x.connect("ode", "aero", r"$u, v, w$")
x.connect("aero", "forces", [r"$D, L, Y$", r"$(drag, lift, side)$"], stack=True)
x.connect("ode", "forces", [r"$T_x, T_y, T_z$", r"$\phi, \theta, \psi$"])
x.connect("forces", "eom", r"$F_x, F_y, F_z$")
x.connect("ode", "eom", [r"$m, J_{xx}, J_{yy}, J_{zz}$", r"$p, q, r$", r"$l_x, l_y, l_z$"])

# EOM outputs (derivatives)
x.connect("eom", "ode", [r"$\dot{u}, \dot{v}, \dot{w}$", 
                          r"$\dot{\phi}, \dot{\theta}, \dot{\psi}$", 
                          r"$\dot{x}, \dot{y}, \dot{z}$"])

# ODE back to phase (defects)
x.connect("ode", "climb", r"defects")
x.connect("ode", "cruise", r"defects")
x.connect("ode", "descent", r"defects")

# Phase continuity constraints
x.connect("climb", "cruise", [r"$t, u, v, w$", r"$\phi, \theta, \psi$", r"$x, y, z$"])
x.connect("cruise", "descent", [r"$t, u, v, w$", r"$\phi, \theta, \psi$", r"$x, y, z$"])

# Phases back to optimizer (constraints and objective)
x.connect("climb", "opt", [r"$z_{final} = 100m$", r"$|T| \leq 19.62N$"])
x.connect("cruise", "opt", [r"$z_{final} = 100m$", r"$x \leq 1500m$", r"$|T| \leq 19.62N$"])
x.connect("descent", "opt", [r"$z_{final} = 0m$", r"$|T| \leq 19.62N$", r"$J = t_{final}$"])

# Add process arrow
x.add_process(["opt", "climb", "ode", "atmos", "aero", "forces", "eom", "cruise", "descent", "opt"], arrow=True)

# Write output
x.write("vtol_xdsm", cleanup=False)
x.write_sys_specs("vtol_xdsm_specs")

print("XDSM diagram generated successfully!")
print("Output files:")
print("  - vtol_xdsm.tex")
print("  - vtol_xdsm.pdf (if LaTeX is available)")
print("  - vtol_xdsm_specs.json")
print("\nTo generate PDF manually, run:")
print("  pdflatex vtol_xdsm.tex")