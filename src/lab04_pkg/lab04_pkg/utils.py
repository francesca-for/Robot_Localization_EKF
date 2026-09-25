import sympy
sympy.init_printing(use_latex='mathjax')
from sympy import symbols, Matrix, latex
import numpy as np
from math import sin, cos, degrees


import sympy
import numpy as np

def eval_jacobiane_Gt_e_Vt():
    x, y, theta, v, w, dt = symbols('x y theta v w dt')
    R = v / w
    beta = theta + w * dt
    gux = Matrix(
        [
            [x - R * sympy.sin(theta) + R * sympy.sin(beta)],
            [y + R * sympy.cos(theta) - R * sympy.cos(beta)],
            [beta],
        ]
    )
    Gt = gux.jacobian(Matrix([x, y, theta]))
    Vt = gux.jacobian(Matrix([v, w]))
    
    # Crea le funzioni lambdify che accettano (x, y, theta, v, w, dt)
    eval_Gt_sym = sympy.lambdify((x, y, theta, v, w, dt), Gt, "numpy")
    eval_Vt_sym = sympy.lambdify((x, y, theta, v, w, dt), Vt, "numpy")
    
    def eval_Gt(*args):
        x, y, theta, v, w, dt = args
        
        # **FORZA w a essere sempre != 0**
        w_safe = w if abs(w) >= 1e-6 else (1e-6 if w >= 0 else -1e-6)
        
        return eval_Gt_sym(x, y, theta, v, w_safe, dt)

    def eval_Vt(*args):
        x, y, theta, v, w, dt = args
        
        # **FORZA w a essere sempre != 0**
        w_safe = w if abs(w) >= 1e-6 else (1e-6 if w >= 0 else -1e-6)
        
        return eval_Vt_sym(x, y, theta, v, w_safe, dt)
    
    return eval_Gt, eval_Vt
    
def sample_velocity_motion_model(x, u, sigma_u, g_extra_args):
    """
    x: [x, y, theta]
    u: [v, w] 
    sigma_u: [sigma_v, sigma_w]
    g_extra_args: (dt,) - tupla con dt
    """
    # Estrai dt
    if isinstance(g_extra_args, (tuple, list)) and len(g_extra_args) > 0:
        dt = g_extra_args[0]
    else:
        dt = 0.05
        
    # Aggiungi rumore
    v_hat = u[0] #+ np.random.normal(0.0, sigma_u[0])   # media e std dev
    w_hat = u[1] #+ np.random.normal(0.0, sigma_u[1])

    # Applica modello di moto con gestione divisione zero
    if abs(w_hat) < 1e-6:
        x_prime = x[0] + v_hat * dt * np.cos(x[2])
        y_prime = x[1] + v_hat * dt * np.sin(x[2])
        theta_prime = x[2]
    else:
        r = v_hat / w_hat
        x_prime = x[0] - r * np.sin(x[2]) + r * np.sin(x[2] + w_hat * dt)
        y_prime = x[1] + r * np.cos(x[2]) - r * np.cos(x[2] + w_hat * dt)
        theta_prime = x[2] + w_hat * dt

    theta_prime = normalize_angle(theta_prime)

    return np.array([x_prime, y_prime, theta_prime])

def eval_jacobian_hux_Ht():
    mx, my = symbols("m_x m_y")
    x,y,theta= symbols("x y theta")
    hx = Matrix(
        [
            [sympy.sqrt((mx - x) ** 2 + (my - y) ** 2)],
            [sympy.atan2(my - y, mx - x) - theta],
        ]
    )
    eval_hx = sympy.lambdify((x, y, theta, mx, my), hx, "numpy")
    Ht = hx.jacobian(Matrix([x, y, theta]))
    eval_Ht = sympy.lambdify((x, y, theta, mx, my), Ht, "numpy")

    return eval_hx, eval_Ht


def normalize_angle(angle):
    # Normalizza l'angolo tra -pi e pi
    return (angle + np.pi) % (2 * np.pi) - np.pi


def residual_measurement(z, z_hat):
    y = z - z_hat
    
    # Gestione sia per array 1D che 2D
    if len(y.shape) > 1:
        y[1, 0] = normalize_angle(y[1, 0])
    else:
        y[1] = normalize_angle(y[1])
        
    return y

