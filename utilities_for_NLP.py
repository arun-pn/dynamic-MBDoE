# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 18:55:22 2020

@author: p_aru
"""

import numpy as np
import casadi as cad
from utilities import construct_polynomial_basis, FIM_t

def construct_NLP_collocation(f, x_0, xp_0, lbx_0, ubx_0, lbx, ubx, lbu, ubu, lbtheta, ubtheta, lbxp, ubxp, ubT, N, nsp, mg):
    n_x = np.shape(x_0)[0]
    n_u = np.shape(lbu)[0]
    n_theta = np.shape(lbtheta)[0]
    
    # Degree of interpolating polynomial
    d = 4
    C, D, B = construct_polynomial_basis(d, 'legendre')
    
    # Start with an empty NLP
    w = []  # for decision vector
    w0 = []     # for initial guess of decision vector
    lbw = []    # lower bound for decision vector
    ubw = []    # upper bound for decision vector
    J = 0   # objective function value
    g = []  # vector of equality constraints
    lbg = []    # lower bound for constraints
    ubg = []    # upper bound for constraints

    # For plotting x and u given w
    x_plot = []
    x_plotp = []
    u_plot = []
    discrete = []
    
    # "Lift" initial conditions
    Xk = cad.SX.sym('X0', n_x)
    w += [Xk]
    Xpk = cad.SX.sym('Xp0', n_x * n_theta)
    w += [Xpk]

    lbw.extend(lbx_0)
    ubw.extend(ubx_0)
    w0.extend(x_0)
    discrete += [False] * n_x

    x_plot += [Xk]

    lbw.extend(xp_0)
    ubw.extend(xp_0)
    w0.extend(xp_0)
    discrete += [False] * (n_x * n_theta)

    x_plotp += [Xpk]
    
    thetak = cad.SX.sym('theta', np.shape(lbtheta)[0])
    w += [thetak]

    count = np.zeros(N + 1)

    lbw.extend(lbtheta)
    ubw.extend(ubtheta)

    w0.extend(lbtheta)
    discrete += [False] * n_theta


    j1 = 0

    T = cad.SX.sym('T')
    w += [T]
    lbw.extend([mg * (nsp - 1)])
#    lbw.extend([0.1])
    ubw.extend([ubT])
    w0.extend([ubT])
    discrete += [False]

    ss = 0
    b = cad.SX.sym('b', N)
    w += [b]
    lbw.extend([0.] * N)
    ubw.extend([1.] * N)
#    w0.extend([0.] * N)
    w0.extend([0.] * (N - nsp))
    w0.extend([1.] * nsp)
    discrete += [False] * N
    
    # Formulate the NLP
    for k in range(N):
        # New NLP variable for the control
        Uk = cad.SX.sym('U_' + str(k), n_u)
        w += [Uk]
        lbw.extend(lbu)
        ubw.extend(ubu)
        w0.extend(ubu)
        discrete += [False] * n_u

        u_plot += [Uk]
        # -------------------
        DTk = cad.SX.sym('DT_' + str(k))
        w += [DTk]
        lbw += [0.001]
        ubw += [ubT] # or use 20
        w0 += [3.5]
        discrete += [False]

        #    Ts += [DTk]
        h = DTk
        ss += DTk
        # --------------------
        # State at collocation points
        Xc = []
        Xpc = []
        for j in range(d):
            Xkj = cad.SX.sym('X_' + str(k) + '_' + str(j), n_x)
            Xc += [Xkj]
            w += [Xkj]
            lbw.extend(lbx)
            ubw.extend(ubx)
            w0.extend([0] * n_x)
            discrete += [False] * n_x
            
        for j in range(d):
            Xpkj = cad.SX.sym('Xp_' + str(k) + '_' + str(j), n_x * n_theta)
            Xpc += [Xpkj]
            w += [Xpkj]
            lbw.extend(lbxp)
            ubw.extend(ubxp)
            w0.extend([0] * (n_x * n_theta))
            discrete += [False] * (n_x * n_theta)
            
        # Loop over collocation points
        Xk_end = D[0] * Xk
        Xpk_end = D[0] * Xpk
        for j in range(1, d + 1):
            # Expression for the state derivative at the collocation point
            xp = C[0, j] * Xk
            xpp = C[0, j] * Xpk
            for r in range(d):
                xp = xp + C[r + 1, j] * Xc[r]
                xpp = xpp + C[r + 1, j] * Xpc[r]
            
            # Append collocation equations
            fj, qj, dxpj = f(Xc[j - 1], Uk, thetak, Xpc[j - 1])
            g += [(h * fj - xp)]
            lbg.extend([0] * n_x)
            ubg.extend([0] * n_x)

            g += [(h * dxpj - xpp)]
            lbg.extend([0] * (n_x * n_theta))
            ubg.extend([0] * (n_x * n_theta))
            
            # Add contribution to the end state
            Xk_end = Xk_end + D[j] * Xc[j - 1]

            Xpk_end = Xpk_end + D[j] * Xpc[j - 1]
#            if int(j1) < np.shape(t_meas)[0]:
#                if np.real(k * T / N) == t_meas[j1]:
#                    count[k] = 1
#                    j1 += 1
            # Add contribution to quadrature function
        #      J = J + B[j]*qj*h
        
        # New NLP variable for state at end of interval
        Xk = cad.SX.sym('X_' + str(k + 1), n_x)
        w += [Xk]
#        lbw.extend([-np.inf, -np.inf]) # why not lbx?
        lbw.extend(lbx) # why not lbx?
        ubw.extend(ubx)
        w0.extend([0] * n_x)
        discrete += [False] * n_x

        x_plot += [Xk]
        Xpk = cad.SX.sym('Xp_' + str(k + 1), n_x * n_theta)
        w += [Xpk]
        lbw.extend([-np.inf] * (n_x * n_theta))
        ubw.extend([np.inf] * (n_x * n_theta))
        w0.extend([0] * (n_x * n_theta))
        discrete += [False] * (n_x * n_theta)

        x_plotp += [Xpk]
        # Add equality constraint
        g += [Xk_end - Xk]
        lbg.extend([0] * n_x)
        ubg.extend([0] * n_x)

        g += [Xpk_end - Xpk]
        lbg.extend([-0.00001] * (n_x * n_theta))
        ubg.extend([0.00001] * (n_x * n_theta))
        g += [DTk-T/N]
        lbg.extend([0])
        ubg.extend([0])
        
        # add the inequality constraint for sampling
        
#        if k > 0:
#            g += [((b[k] * (T/N) * (k+1) - b[k-1] * (T/N) * k)**2) - (0.5**2) * b[k] * b[k-1]]
#            lbg.extend([0.0])
#            ubg.extend([np.inf])
            
        for i in range(N):
            if i != k:
                g += [((b[k] * (T/N) * (k+1) - b[i] * (T/N) * (i+1))**2) - (mg**2) * b[k] * b[i]]
                lbg.extend([0.0])
                ubg.extend([np.inf])
                
    
#    for k in range(1,N):
#        g += [((b[k] * (T/N) * (k+1) - b[k-1] * (T/N) * k)**2) - (0.5**2) * b[k] * b[k-1]]
#        lbg.extend([0.0])
#        ubg.extend([np.inf])
    
        
    count[-1] = 1.
    # Concatenate vectors
    # w = vertcat(*w)
    # g = vertcat(*g)
    # x_plot = horzcat(*x_plot)
    # u_plot = horzcat(*u_plot)
    # w0 = np.concatenate(w0)
    # lbw = np.concatenate(lbw)
    # ubw = np.concatenate(ubw)
    # lbg = np.concatenate(lbg)
    # ubg = np.concatenate(ubg)
    g += [ss-T]
    lbg.extend([0])
    ubg.extend([0])
    
    # sampling constraint in the first control interval (finite element)
    g += [((b[0] * (T/N) - 0.0)**2) - (mg**2) * b[0]]
    lbg.extend([0])
    ubg.extend([np.inf])

    g += [cad.sum1(b[:])]
    lbg.extend([nsp])
    ubg.extend([nsp])
    
    # Create an NLP solver
#    mle = maximum_likelihood_est(count, w[3 + 2 * d + 2::2 + 2 * d + 2], x_meas, [1, 5], N)
    mle = FIM_t(x_plotp, b, 'A_optimal')
    problem = {'f': mle, 'x': cad.vertcat(*w), 'g': cad.vertcat(*g)}
    trajectories = cad.Function('trajectories', [cad.vertcat(*w)]
                           , [cad.horzcat(*x_plot), cad.horzcat(*u_plot), cad.horzcat(*x_plotp), cad.horzcat(T), cad.horzcat(b)], ['w'], ['x', 'u', 'xp', 'T','b'])

    return problem, w0, lbw, ubw, lbg, ubg, trajectories, discrete