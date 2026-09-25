import sympy
sympy.init_printing(use_latex='mathjax')
from sympy import symbols, Matrix, latex
import numpy as np
from math import sin, cos, degrees

import sympy
import numpy as np

# 3. Funzioni H e h (definite localmente o in utils)
# h(x) -> restituisce [v, w] dallo stato
def hx_odom(x, y, theta, v, w):
    return np.array([v, w])
        
# H(x) -> matrice 2x5 che seleziona v e w
def Ht_odom(x, y, theta, v, w):
    H = np.zeros((2, 5))
    H[0, 3] = 1 # v
    H[1, 4] = 1 # w
    return H

# 3. Funzioni H e h
def hx_imu(x, y, theta, v, w):
    return np.array([w])
        
def Ht_imu(x, y, theta, v, w):
    H = np.zeros((1, 5))
    H[0, 4] = 1 # w (ultimo elemento)
    return H




def eval_jacobiane_Gt_e_Vt():
    x, y, theta, v, w, v_in, w_in, dt = symbols('x y theta v w v_in w_in dt')
    R = v / w
    beta = theta + w * dt
    gux = Matrix(
        [
            [x - R * sympy.sin(theta) + R * sympy.sin(beta)],
            [y + R * sympy.cos(theta) - R * sympy.cos(beta)],
            [beta],
            [v],
            [w],
        ]
    )
    Gt = gux.jacobian(Matrix([x, y, theta, v, w]))
    #Vt = gux.jacobian(Matrix([v, w]))
    Vt = Matrix([
        [0, 0],
        [0, 0],
        [0, 0],
        [1, 0], 
        [0, 1]
    ])
    
    # Crea le funzioni lambdify che accettano (x, y, theta, v, w, dt)
    eval_Gt_sym = sympy.lambdify((x, y, theta, v, w, v_in, w_in, dt), Gt, "numpy")
    eval_Vt_sym = sympy.lambdify((x, y, theta, v, w, v_in, w_in, dt), Vt, "numpy")

    def eval_Gt(*args):
        x, y, theta, v, w, _, _, dt = args
        
        # **FORZA w a essere sempre != 0**
        w_safe = w if abs(w) >= 1e-6 else (1e-6 if w >= 0 else -1e-6)

        return eval_Gt_sym(x, y, theta, v, w_safe, v, w_safe, dt)

    def eval_Vt(*args):
        x, y, theta, v, w, _, _, dt = args
        
        # **FORZA w a essere sempre != 0**
        w_safe = w if abs(w) >= 1e-6 else (1e-6 if w >= 0 else -1e-6)
        
        return eval_Vt_sym(x, y, theta, v, w_safe, v, w_safe, dt)
    
    return eval_Gt, eval_Vt
    
def sample_velocity_motion_model(x, u, sigma_u, g_extra_args):
    """
    x: [x, y, theta, v, w]
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
    v_hat = x[3] #+ np.random.normal(0.0, sigma_u[0])   # media e std dev
    w_hat = x[4] #+ np.random.normal(0.0, sigma_u[1])

    # Applica modello di moto con gestione divisione zero
    if abs(w_hat) < 1e-6:
        x_prime = x[0] + v_hat * dt * np.cos(x[2])
        y_prime = x[1] + v_hat * dt * np.sin(x[2])
        theta_prime = x[2]
        v_prime = v_hat
        w_prime = w_hat
    else:
        r = v_hat / w_hat
        x_prime = x[0] - r * np.sin(x[2]) + r * np.sin(x[2] + w_hat * dt)
        y_prime = x[1] + r * np.cos(x[2]) - r * np.cos(x[2] + w_hat * dt)
        theta_prime = x[2] + w_hat * dt
        v_prime = v_hat
        w_prime = w_hat

    theta_prime = normalize_angle(theta_prime)

    return np.array([x_prime, y_prime, theta_prime, v_prime, w_prime])


def eval_jacobian_hux_Ht():
    mx, my = symbols("m_x m_y")
    x,y,theta,v,w = symbols("x y theta v w")
    hx = Matrix(
        [
            [sympy.sqrt((mx - x) ** 2 + (my - y) ** 2)],
            [sympy.atan2(my - y, mx - x) - theta],
        ]
    )
    eval_hx = sympy.lambdify((x, y, theta, v, w, mx, my), hx, "numpy")
    Ht = hx.jacobian(Matrix([x, y, theta, v, w]))
    eval_Ht = sympy.lambdify((x, y, theta, v, w, mx, my), Ht, "numpy")

    return eval_hx, eval_Ht

def eval_jacobian_Hx_odom(state):
    v = state[3]
    w = state[4]
    return np.array([v, w])


def normalize_angle(angle):
    # Normalizza l'angolo tra -pi e pi
    return (angle + np.pi) % (2 * np.pi) - np.pi


def residual_measurement(z, z_hat):
    # Calcola la differenza z - z_hat.
    # Normalizza l'angolo per la seconda componente (bearing).
    # z e z_hat sono array (2, 1) o (2,).
    y = z - z_hat
    # La componente 0 è il range (distanza), non va toccata.
    # La componente 1 è il bearing (angolo), va normalizzato.
    
    # Gestione sia per array 1D che 2D
    if len(y.shape) > 1:
        y[1, 0] = normalize_angle(y[1, 0])
    else:
        y[1] = normalize_angle(y[1])
        
    return y