# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 14:13:40 2020

@author: p_aru
"""

import numpy as np
import casadi as cad
import pyomo.environ as py
import pyomo.dae as pyd
from scipy.integrate import odeint
import math
import scipy.stats as stats

def dynamicmodel():
    '''
    Define the model that is meant to describe the physical system.
    Return model f with domain: [x,u,theta,Q] & codomain: [xdot,L,Qdot]
    f: [x,u,theta,Q] ---> [xdot,L,Qdot]
    '''
    n_x = 2
    n_u = 2
    n_theta = 4
    x1 = cad.SX.sym('x1')
    x2 = cad.SX.sym('x2')
    x = cad.vertcat(x1, x2)
    u1 = cad.SX.sym('u1')
    u2 = cad.SX.sym('u2')
    u = cad.vertcat(u1, u2)
    theta1 = cad.SX.sym('theta1')
    theta2 = cad.SX.sym('theta2')
    theta3 = cad.SX.sym('theta3')
    theta4 = cad.SX.sym('theta4')
    theta = cad.vertcat(theta1, theta2, theta3, theta4)
    
    # bakers example
    r = (theta1 * x2 / (theta2 + x2))
    xdot = cad.vertcat(((r - u1 - theta4) * x1),
                   ((-r * x1 / theta3) + u1 * (u2 - x2)))
    # Calculate on the fly dynamic sensitivities without the need of perturbations
    x_p = cad.SX.sym('xp', np.shape(x)[0] * np.shape(theta)[0])
    xpdot = []
    for i in range(4):
        xpdot = cad.vertcat(xpdot, cad.jacobian(xdot, x) @ (x_p[n_x * i: n_x * i + n_x])
                        + cad.jacobian(xdot, theta)[n_x * i: n_x * i + n_x])

    # Quadrature
    L = []  # x1 ** 2 + x2 ** 2 + 1*u1 ** 2 + 1*u2**2
    # Algebraic
    alg = []

    # Continuous time dynamics
    f = cad.Function('f', [x, u, theta, x_p], [xdot, L, xpdot], ['x', 'u', 'theta', 'xp'], ['xdot', 'L', 'xpdot'])

    return f, n_x, n_u, n_theta
    
def integratorfunction(f, n_x, n_u, n_theta):
    '''
    This function uses a simple fixed step RK4 method to obtain 
    to integrate the model equations.
    Returns F: [x,u,dt,theta] ---> [xf,obj]
    '''
    M = 4  # RK4 steps per interval
    DT = cad.SX.sym('DT')
    DT1 = DT / M
    X0 = cad.SX.sym('X0', n_x)   # differential states
    U = cad.SX.sym('U', n_u)     # controls
    theta = cad.SX.sym('theta', n_theta)     # parameters
    xp0 = cad.SX.sym('xp', np.shape(X0)[0] * np.shape(theta)[0])    # sensitivity
    X = X0
    Q = 0
    G = 0
    S = xp0
    for j in range(M):
        k1, k1_q, k1_a, k1_p = f(X, U, theta, S)
        k2, k2_q, k2_a, k2_p = f(X + DT1 / 2 * k1, U, theta, S + DT1 / 2 * k1_p)
        k3, k3_q, k3_a, k3_p = f(X + DT1 / 2 * k2, U, theta, S + DT1 / 2 * k2_p)
        k4, k4_q, k4_a, k4_p = f(X + DT1 * k3, U, theta, S + DT1 * k3_p)
        X = X + DT1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        Q = Q + DT1 / 6 * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
        G = G + DT1 / 6 * (k1_a + 2 * k2_a + 2 * k3_a + k4_a)
        S = S + DT1 / 6 * (k1_p + 2 * k2_p + 2 * k3_p + k4_p)
    F = Function('F', [X0, U, DT, theta, xp0], [X, Q, G, S], ['x0', 'p', 'DT', 'theta', 'xp0'], ['xf', 'qf', 'g', 'xp'])

    return F

def FIM_t(xpdot, b, criterion):
    '''
    computes FIM at a given time.
    b is the bonary variable which selects or not the time point
    '''
    n_x = 2
    n_theta = 4
    FIM_sample = np.zeros([n_theta,n_theta])
    FIM_0 = cad.inv((((10.0 - 0.001)**2)/12) * cad.SX.eye(n_theta))
    FIM_sample += FIM_0
    for i in range(np.shape(xpdot)[0]-1):
        xp_r = cad.reshape(xpdot[i+1], (n_x, n_theta))
#    vv = np.zeros([ntheta[0], ntheta[0], 1 + N])
#    for i in range(0, 1 + N):
        FIM_sample += b[i] * xp_r.T @ np.linalg.inv(np.array([[0.01, 0], [0, 0.05]])) @ xp_r# + np.linalg.inv(np.array([[0.01, 0, 0, 0], [0, 0.05, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0.2]]))
#    FIM  = solve(FIM1, SX.eye(FIM1.size1()))
 #   [Q, R] = qr(FIM1.expand())
     
    if criterion == 'D_optimal':
#        objective = -cad.log(cad.det(FIM_sample) + 1e-10)
        objective = -2 * cad.sum1(cad.log(cad.diag(cad.chol(FIM_sample)))) # by Cholesky factorisation
#        objective = -cad.log((cad.det(FIM_sample)**2))
#        objective = -cad.det(FIM_sample)
    elif criterion == 'A_optimal':
        objective = -cad.log(cad.trace(FIM_sample) + 1e-10)
    return objective

def construct_polynomial_basis(d, root):
    # Get collocation points
    tau_root = np.append(0, cad.collocation_points(d, root))

    # Coefficients of the collocation equation
    C = np.zeros((d + 1, d + 1))

    # Coefficients of the continuity equation
    D = np.zeros(d + 1)

    # Coefficients of the quadrature function
    B = np.zeros(d + 1)

    # Construct polynomial basis
    for j in range(d + 1):
        # Construct Lagrange polynomials to get the polynomial basis at the collocation point
        p = np.poly1d([1])
        for r in range(d + 1):
            if r != j:
                p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j] - tau_root[r])

        # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
        D[j] = p(1.0)
        # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the collocation
        # equation
        pder = np.polyder(p)
        for r in range(d + 1):
            C[j, r] = pder(tau_root[r])

        # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
        pint = np.polyint(p)
        B[j] = pint(1.0)

    return C, D, B


def fermentation(x,t,u,theta):
    x1 = x[0]                                   
    x2 = x[1]
    r = ((theta[0] * x2)/(theta[1] + x2))
    dx1dt = (r - u[0] - theta[3]) * x1
    dx2dt = ((-r * x1)/theta[2]) + (u[0] * (u[1] - x2))
    return [dx1dt, dx2dt]

def dynsim(y0,u,theta,T,N):
    x0 = []
    x0 += [y0]
    y_hat = []
    for i in range(N):
        t = np.array([(T/N) * i, (T/N) * (i+1)])
        y_hat += [odeint(fermentation,x0[-1],t,args=(u[:,i],theta))]
        x0 += [y_hat[-1][-1]]
    return np.array(x0)

def insilicoexp(y0,u,truetheta,T,N,nm,sigma_y,tmeas):
    x0 = []
    x0 += [y0]
    y_hat = []
    ys = []
    for i in range(N):
        t = np.array([(T/N) * i, (T/N) * (i+1)])
        y_hat += [odeint(fermentation,x0[-1],t,args=(u[:,i],truetheta))]
        x0 += [y_hat[-1][-1]]
    for i in range(np.shape(np.where(tmeas > 0.99)[0])[0]):
        ys += [np.array(x0)[0:][np.where(tmeas > 0.99)[0][i]]]
    error = np.random.multivariate_normal(np.array([0] * np.shape(sigma_y)[0]), sigma_y, nm)
    ym = np.array(ys)[0:nm] + error
    return ym
    
def dynsen(y0,u,theta,T,N,tmeas,nm):
    epsilon = 0.001
    p_matrix = np.zeros([np.shape(theta)[0]+1,np.shape(theta)[0]])
    for i in range(np.shape(theta)[0]+1):
        p_matrix[i] = theta
    for i in range(np.shape(theta)[0]):
        p_matrix[i][i] = theta[i] * (1 + epsilon)
    x0 = []
    y_hat = []
    sen = np.zeros([N+1,np.shape(theta)[0],np.shape(y0)[0]])
    obs_sen = []
    for theta in p_matrix:
        x0 += [y0]
        for i in range(N):
            t = np.array([(T/N) * i, (T/N) * (i+1)])
            y_hat += [odeint(fermentation,x0[-1],t,args=(u[:,i],theta))]
            x0 += [y_hat[-1][-1]]
    for i in range(N + 1):
        for j in range(np.shape(theta)[0]):
            sen[i][j] = (np.array(x0)[j * (N + 1) + i] - np.array(x0)[np.shape(theta)[0] * (N + 1) + i])/(epsilon * theta[j])
    for i in range(np.shape(np.where(tmeas > 0.99)[0])[0]):
        obs_sen += [np.array(sen)[1:][np.where(tmeas > 0.99)[0][i]]]
    act_sen = np.array(obs_sen)[0:nm]
    return act_sen

def lrf(y0,u,theta,T,N,nm,sigma_y,tmeas,xmeas):
    x0 = []
    x0 += [y0]
    y_hat = []
    ys = []
    for i in range(N):
        t = np.array([(T/N) * i, (T/N) * (i+1)])
        y_hat += [odeint(fermentation,x0[-1],t,args=(u[:,i],theta))]
        x0 += [y_hat[-1][-1]]
    for i in range(np.shape(np.where(tmeas > 0.99)[0])[0]):
        ys += [np.array(x0)[1:][np.where(tmeas > 0.99)[0][i]]]
    wr = sum(np.linalg.inv(sigma_y) @ sum((xmeas - np.array(ys))**2))
    nllf = 0.5 * nm * np.shape(xmeas)[1] * math.log(2 * math.pi) + 0.5 * nm * sum(np.log(np.diagonal(sigma_y))) + 0.5 * wr
    return nllf

def lrtest(y0,u,theta0,thetahat,T,N,nm,sigma_y,tmeas,xmeas):
    nllf_theta0 = lrf(y0,u,theta0,T,N,nm,sigma_y,tmeas,xmeas)
    nllf_thetahat = lrf(y0,u,thetahat,T,N,nm,sigma_y,tmeas,xmeas)
    lrst = 2 * (nllf_theta0 - nllf_thetahat)
    sfvalue = stats.distributions.chi2.sf(lrst, np.shape(theta0)[0])
    pvalue = 1 - stats.chi2.cdf(lrst, np.shape(theta0)[0])
    ref_chisquare = stats.chi2.ppf((1 - 0.05),np.shape(theta0)[0])
    return sfvalue


def optkm1(y0, u, ymeas, td, sigma, tmeas, ig, lb, ub, T, N, scheme_name):
    model = py.ConcreteModel()
#    ft = int(T)
    ft = (T)
    total_elements = int(N)
    element_length = (ft/total_elements)
    
    scheme_name = 'LAGRANGE-LEGENDRE'
    scheme_name = 'LAGRANGE-RADAU'
    
    d = 4
    
    model.t = pyd.ContinuousSet(initialize = np.linspace(ft/total_elements, ft, total_elements).tolist(), bounds = (0,ft))
#    model.t = pyd.ContinuousSet(initialize = sorted(model.tau), bounds = (0,ft))
    
    
    # Data: Experient 1
    model.meas_t_1 = py.Set(initialize = tmeas)#, within = model.t)
    model.tk = py.Set(initialize = td)#, within = model.t)
    model.x1_1_meas = py.Param(model.meas_t_1, initialize = {tmeas[k]:ymeas[0][k] for k in range(np.shape(tmeas)[0])})
    model.x2_1_meas = py.Param(model.meas_t_1, initialize = {tmeas[k]:ymeas[1][k] for k in range(np.shape(tmeas)[0])})
    
    uk = np.zeros([2, np.shape(td)[0]])
    for i in range(total_elements):
        uk[0][(d + 1) * i + 1 : (d + 1) * (i + 1) + 1] = [u[0][i]] * (d + 1) # check the float here. A float is coomin under the function.
        uk[1][(d + 1) * i + 1 : (d + 1) * (i + 1) + 1] = [u[1][i]] * (d + 1)
    
    model.u1 = py.Param(model.tk, initialize = {td[k]:uk[0][k] for k in range(np.shape(td)[0])})
    model.u2 = py.Param(model.tk, initialize = {td[k]:uk[1][k] for k in range(np.shape(td)[0])})
    
    
    model.theta_idx = py.Set(initialize = list(range(np.shape(ig)[0]))) 
    theta_lb_dict = dict(list(enumerate(lb)))
    theta_ub_dict = dict(list(enumerate(ub)))
    theta_dict = dict(list(enumerate(ig)))

    def parmbounds(model,i):
        return (theta_lb_dict[i], theta_ub_dict[i])
    
    # declare model parameters
    model.theta = py.Var(model.theta_idx, initialize = theta_dict, bounds = parmbounds)
    
    # declare differential variables
    model.x1 = py.Var(model.t, within = py.NonNegativeReals, bounds = (0,np.inf))
    model.x2 = py.Var(model.t, within = py.NonNegativeReals, bounds = (0,np.inf))
    
    # declare differential variables for batch case
    n_exp = 3
    expv = list(range(n_exp))
    model.x1 = py.Var(expv, model.t, within = py.NonNegativeReals, bounds = (0,np.inf))
    model.x2 = py.Var(expv, model.t, within = py.NonNegativeReals, bounds = (0,np.inf))

    
    # declare derivatives
    model.dx1dt = pyd.DerivativeVar(model.x1, wrt = model.t)
    model.dx2dt = pyd.DerivativeVar(model.x2, wrt = model.t)
    
    # declare derivatives for batch case
    model.dx1dt = pyd.DerivativeVar(expv, model.x1, wrt = model.t)
    model.dx2dt = pyd.DerivativeVar(expv, model.x2, wrt = model.t)
    
    def diffeqn1(model,t):
        if model.t == 0:
            return py.Constraint.Skip
        return model.dx1dt[t] == ((model.theta[0] * model.x2[t] / (model.theta[1] + model.x2[t])) - model.u1[t] - model.theta[3]) * model.x1[t]
    model.x1cons = py.Constraint(model.t, rule = diffeqn1)
    
    def diffeqn2(model,t):
        if model.t == 0:
            return py.Constraint.Skip
        return model.dx2dt[t] == ((-(model.theta[0] * model.x2[t] / (model.theta[1] + model.x2[t])) * model.x1[t]) / model.theta[2]) + model.u1[t] * (model.u2[t] - model.x2[t])
    model.x2cons = py.Constraint(model.t, rule = diffeqn2)

    def init_conditions(model):
        yield model.x1[0] == y0[0]
        yield model.x2[0] == y0[1]
    model.init_cond = py.ConstraintList(rule = init_conditions)
    
    # discretize model using orthogonal collocation
    discretizer = py.TransformationFactory('dae.collocation')
    discretizer.apply_to(model, wrt = model.t, nfe = total_elements, ncp = d, scheme = scheme_name)
    
    def obj_expression(model):
        chisq_1 = sum((((model.x1_1_meas[j] - model.x1[j])**2)/sigma[0]) + (((model.x2_1_meas[j] - model.x2[j])**2)/sigma[1]) for j in model.meas_t_1)
        obj_fun = chisq_1
        return obj_fun
    model.objfun = py.Objective(rule = obj_expression)
    return model

def biofmodel(y0, u, ymeas, sigma, tmeas, ig, lb, ub, T, N):
    model = py.ConcreteModel()
    ft = T
    total_elements = int(N)
    element_length = ft/total_elements
    
    scheme_name = 'LAGRANGE-LEGENDRE'
#    scheme_name = 'LAGRANGE-RADAU'
    discretizer = py.TransformationFactory('dae.collocation')
    
    d = 4
    
    model.tau = pyd.ContinuousSet(bounds = (0,ft))
    discretizer.apply_to(model, wrt = model.tau, nfe = total_elements, ncp = d, scheme = scheme_name)
    model.t = pyd.ContinuousSet(initialize = np.round(np.linspace(ft/total_elements, ft, total_elements), 1).tolist(), bounds = (0,ft))
    
    # dynamic ata
    model.meas_t = py.Set(initialize = tmeas, within = model.t)
    model.x1_meas = py.Param(model.meas_t, initialize = {tmeas[k]:ymeas[0][k] for k in range(np.shape(tmeas)[0])})
    model.x2_meas = py.Param(model.meas_t, initialize = {tmeas[k]:ymeas[1][k] for k in range(np.shape(tmeas)[0])})
    
    
    model.theta_idx = py.Set(initialize = list(range(np.shape(ig)[0]))) 
    theta_lb_dict = dict(list(enumerate(lb)))
    theta_ub_dict = dict(list(enumerate(ub)))
    theta_dict = dict(list(enumerate(ig)))

    def parmbounds(model,i):
        return (theta_lb_dict[i], theta_ub_dict[i])
    # declare model parameters
    model.theta = py.Var(model.theta_idx, initialize = theta_dict, bounds = parmbounds)
    
    model.tprime = py.Set(initialize = sorted(model.tau))
    uk = np.zeros([np.shape(u)[0], np.shape(sorted(model.tau))[0]])
    for i in range(total_elements):
        uk[0][(d + 1) * i + 1 : (d + 1) * (i + 1) + 1] = [u[0][i]] * (d + 1)
        uk[1][(d + 1) * i + 1 : (d + 1) * (i + 1) + 1] = [u[1][i]] * (d + 1)
    model.u1 = py.Param(model.tprime, initialize = {sorted(model.tau)[k]:uk[0][k] for k in range(np.shape(sorted(model.tau))[0])})
    model.u2 = py.Param(model.tprime, initialize = {sorted(model.tau)[k]:uk[1][k] for k in range(np.shape(sorted(model.tau))[0])})
    # declare differential variables
    model.x1 = py.Var(model.t, within = py.NonNegativeReals, bounds = (0,np.inf))
    model.x2 = py.Var(model.t, within = py.NonNegativeReals, bounds = (0,np.inf))

    # declare derivatives
    model.dx1dt = pyd.DerivativeVar(model.x1, wrt = model.t)
    model.dx2dt = pyd.DerivativeVar(model.x2, wrt = model.t)
    
    def diffeqn1(model,t):
        if model.t == 0:
            return py.Constraint.Skip
        return model.dx1dt[t] == ((model.theta[0] * model.x2[t] / (model.theta[1] + model.x2[t])) - model.u1[t] - model.theta[3]) * model.x1[t]
    model.x1cons = py.Constraint(model.t, rule = diffeqn1)
    
    def diffeqn2(model,t):
        if model.t == 0:
            return py.Constraint.Skip
        return model.dx2dt[t] == ((-(model.theta[0] * model.x2[t] / (model.theta[1] + model.x2[t])) * model.x1[t]) / model.theta[2]) + model.u1[t] * (model.u2[t] - model.x2[t])
    model.x2cons = py.Constraint(model.t, rule = diffeqn2)

    def init_conditions(model):
        yield model.x1[0] == y0[0]
        yield model.x2[0] == y0[1]
    model.init_cond = py.ConstraintList(rule = init_conditions)
    
    discretizer.apply_to(model, wrt = model.t, nfe = total_elements, ncp = d, scheme = scheme_name)
    
    def obj_expression(model):
        chisq_1 = sum((((model.x1_meas[j] - model.x1[j])**2)/sigma[0][0]) + (((model.x2_meas[j] - model.x2[j])**2)/sigma[1][1]) for j in model.meas_t)
        obj_fun = chisq_1
        return obj_fun
    model.objfun = py.Objective(rule = obj_expression)
    
    return model