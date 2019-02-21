"""
Module for getting trajectories of typical toy models in nonlinear TSA
======================================================================

"""

# Created: Sun Sep 09, 2018  03:07pm
# Last modified: Thu Dec 20, 2018  04:02pm
# Copyright: Bedartha Goswami <goswami@pik-potsdam.de>


import sys
import numpy as np

import scipy.integrate as spint


# disable dive by zero warnings
np.seterr(divide="ignore")


def roessler(pos0, t, params):
    """
    Integrates the Roessler system of equations with i.c. x0 and for time t.

    Equations:
        xdot = - y - z
        ydot = x + a y
        zdot = b + z (x - c)

    Parameters for screw-type chaos:
        a = 0.15
        b = 0.20
        c = 10.00
    Parameters for funnel-type chaos:
        a = 0.30
        b = 0.40
        c = 7.50
    """
    a, b, c = params
    def _equations(_pos, _t):
        """
        Defines the fundamental equations fo the Roessler system.
        """
        x, y, z = _pos
        xdot = - y - z
        ydot = x + a * y
        zdot = b + z * (x - c)

        return xdot, ydot, zdot

    pos = spint.odeint(_equations, pos0, t)

    return pos


def lorenz(pos0, t, params):
    """
    Integrates the Lorenz system of equations with i.c. x0 and for time t.

    Equations:
        xdot = - s (x - y)
        ydot = - xz + rx - y
        zdot = xy - bz

    Parameters for the famous Lorenz butterfly:
        s = 10
        r = 28
        b = 2.66667
    """
    s, r, b = params
    def _equations(_pos, _t):
        """
        Defines the fundamental equations fo the Roessler system.
        """
        x, y, z = _pos
        xdot = - s * (x - y)
        ydot = - x * z + r * x - y
        zdot = x * y - b * z

        return xdot, ydot, zdot

    pos = spint.odeint(_equations, pos0, t)

    return pos


def logistic(x0, tmax, r):
    """
    Returns the time series for the logistic map for given value of r.
    """
    x = [x0]
    for i in range(1, tmax + 1):
        x.append(r * x[i - 1] * (1. - x[i - 1]))

    return np.array(x)


def henon(x0, tmax, params):
    """
    Returns the time series for the logistic map for given value of r.
    """
    a, b = params
    x0, y0 = x0[0], x0[1]
    x = [x0]
    y = [y0]
    for i in range(1, tmax + 1):
        x.append(1. - a * x[i - 1] ** 2 + y[i - 1])
        y.append(b * x [i - 1])
    x, y = np.array(x), np.array(y)
    pos = np.c_[x, y]

    return pos


def coupled_roessler(pos0, t, params):
    """
    Integrates the Roessler system of equations with i.c. x0 and for time t.

    Equations:
        System 1
        --------
        x1dot = - (1 + nu) y1 - z1
        y1dot = (1 + nu) x1 + a y1 + mu (y2 - y1)
        z1dot = b + z1 (x1 - c)

        System 2
        --------
        x2dot = - (1 - nu) y2 - z2
        y2dot = (1 - nu) x2 + a y2 + mu (y1 - y2)
        z2dot = b + z2 (x2 - c)

    Parameters for phase synchronization:
        a = 0.16
        b = 0.3
        c = 8.5
        nu = 0.02
    Transition to phase synchronization occurs at mu = 0.037

    """
    a, b, c, nu, mu = params

    def _equations(_pos, _t):
        """
        Defines the fundamental equations fo the Roessler system.
        """
        x1, y1, z1, x2, y2, z2 = _pos

        # system 1
        x1dot = - (1 + nu) * y1 - z1 + mu * (x2 - x1)
        y1dot = (1 + nu) * x1 + a * y1
        z1dot = b + z1 * (x1 - c)

        # system 2
        x2dot = - (1 - nu) * y2 - z2 + mu * (x1 - x2)
        y2dot = (1 - nu) * x2 + a * y2
        z2dot = b + z2 * (x2 - c)

        return x1dot, y1dot, z1dot, x2dot, y2dot, z2dot

    pos = spint.odeint(_equations, pos0, t)

    return pos


