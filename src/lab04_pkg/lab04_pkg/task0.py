import matplotlib as mpl
mpl.use('TkAgg') 
import matplotlib.pyplot as plt
from  matplotlib.patches import Arc
import numpy as np
from scipy.stats import norm
from math import cos, sin, degrees, atan2, dist
import sympy
from sympy import symbols, Matrix

arrow = u'$\u2191$'


# ------------------------ Probabilistic velocity-based motion model functions ------------------------

def sample_normal_distribution(sigma_sqrd):
    return 0.5 * np.sum(np.random.default_rng().uniform(-np.sqrt(sigma_sqrd), np.sqrt(sigma_sqrd), 12))


def evaluate_sampling_dist(mu, sigma, n_samples, sample_function, title):
    n_bins = int(np.sqrt(n_samples))
    samples = []

    for i in range(n_samples):
        samples.append(sample_function(mu, sigma))

    print(f"{title}")
    print("%s : mean = %.3f, std_dev = %.3f\n" % ("Normal", np.mean(samples), np.std(samples)))

    count, bins, ignored = plt.hist(samples, n_bins, density=True, alpha=0.8, label='Samples')
    plt.plot(bins, norm(mu, sigma).pdf(bins), linewidth=2, color='r', label='PDF')
    plt.xlim([mu - 5*sigma, mu + 5*sigma])
    plt.title(f"Normal distribution of samples ({title})")
    plt.legend()
    plt.grid()
    plt.savefig(f"gaussian_dist_{title.split()[1]}.pdf")
    plt.show()


def sample_velocity_motion_model(x, u, a, dt):
    """ Sample velocity motion model.
    Arguments:
    x -- pose of the robot before moving [x, y, theta]
    u -- velocity reading obtained from the robot [v, w]
    a -- noise parameters of the motion model [a1, a2, a3, a4, a5, a6]
    dt -- time interval of prediction
    """
    # take linear and angular velocity from the components of our command, then we add a normal noise
    # (that is how we model the error) - normal distribution centered in 0 and with the standard deviation
    # given by our parametrization
    v_hat = u[0] + np.random.normal(0, np.sqrt(a[0]*u[0]**2 + a[1]*u[1]**2))
    w_hat = u[1] + np.random.normal(0, np.sqrt(a[2]*u[0]**2 + a[3]*u[1]**2))
    gamma_hat = np.random.normal(0, np.sqrt(a[4]*u[0]**2 + a[5]*u[1]**2))

    # compute estimate of the new pose
    if abs(w_hat) < 1e-6:
        x_prime = x[0] + v_hat * cos(x[2]) * dt
        y_prime = x[1] + v_hat * sin(x[2]) * dt
        theta_prime = x[2] + w_hat * dt + gamma_hat * dt
    else:
        r = v_hat / w_hat
        x_prime = x[0] - r*sin(x[2]) + r*sin(x[2] + w_hat*dt)
        y_prime = x[1] + r*cos(x[2]) - r*cos(x[2] + w_hat*dt)
        theta_prime = x[2] + w_hat*dt + gamma_hat*dt

    return np.array([x_prime, y_prime, theta_prime])


def compute_jacobians_vel_model(x_, u_, dt_):

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

    eval_gux = sympy.lambdify((x, y, theta, v, w, dt), gux, 'numpy')

    Gt = gux.jacobian(Matrix([x, y, theta]))  # Jacobian w.r.t state
    print(sympy.latex(Gt))  # to get latex code for the report
    eval_Gt = sympy.lambdify((x, y, theta, v, w, dt), Gt, "numpy")
    Gt = eval_Gt(x_[0], x_[1], x_[2], u_[0], u_[1], dt_)
    
    Vt = gux.jacobian(Matrix([v, w]))    # Jacobian w.r.t command
    print(sympy.latex(Vt))  # to get latex code for the report
    eval_Vt = sympy.lambdify((x, y, theta, v, w, dt), Vt, "numpy")
    Vt = eval_Vt(x_[0], x_[1], x_[2], u_[0], u_[1], dt_)
    
    return Gt, Vt


def compute_jacobian_sensor_model(x_, m_):
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
    print(sympy.latex(Ht))  # to get latex code for the report
    eval_Ht = sympy.lambdify((x, y, theta, mx, my), Ht, "numpy")
    Ht = eval_Ht(x_[0], x_[1], x_[2], m_[0], m_[1])

    return Ht


# ------------------------ Probabilistic measurement model functions ------------------------


def landmark_range_bearing_sensor(robot_pose, landmark, sigma, max_range=6.0, fov=np.pi/2):
    """""
    Simulate the detection of a landmark with a virtual sensor able to estimate range and bearing
    """""
    m_x, m_y = landmark[:]
    x, y, theta = robot_pose[:]

    r_ = dist([x, y], [m_x, m_y]) + np.random.normal(0., sigma[0])
    phi_ = atan2(m_y - y, m_x - x) - theta + np.random.normal(0., sigma[1])

    # filter z for a more realistic sensor simulation (add a max range distance and a FOV)
    if r_ > max_range or abs(phi_) > fov / 2:
        return None

    return [r_, phi_]


def gaussian(x, mu, sigma):
    return (1.0 / (np.sqrt(2*np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma)**2)


# Normalized Gaussian pdf
def compute_p_hit_dist(dist, max_dist, sigma):
    '''
    Compute the hit probability p_hit for a given distance measurement.
    Args:
        dist: observed distance measurement
        max_dist: maximum measurable distance
        sigma: standard deviation of the Gaussian noise
    Returns:
        p_hit: normalized hit probability
    '''
    # Normalize the Gaussian over [0, max_dist]
    normalize_hit = 1e-9
    for j in range(round(max_dist)):
        normalize_hit += gaussian(j, 0., sigma)
    normalize_hit = 1. / normalize_hit

    p_hit = gaussian(dist, 0., sigma)*normalize_hit

    return p_hit


def landmark_model_prob(z, landmark, robot_pose, max_range, fov, sigma):
    """""
    Landmark sensor model algorithm:
    Inputs:
      - z: the measurements features (range and bearing of the landmark from the sensor) [r, phi]
      - landmark: the landmark position in the map [m_x, m_y]
      - x: the robot pose [x,y,theta]
    Outputs:
     - p: the probability p(z|x,m) to obtain the measurement z from the state x
        according to the estimated range and bearing
    """""
    m_x, m_y = landmark[:]
    x, y, theta = robot_pose[:]
    sigma_r, sigma_phi = sigma[:]

    r_hat = dist([x, y], [m_x, m_y])
    phi_hat = atan2(m_y - y, m_x - x) - theta
    p = compute_p_hit_dist(z[0] - r_hat, max_range, sigma_r) * compute_p_hit_dist(z[1] - phi_hat, fov/2, sigma_phi)

    return p


def landmark_model_sample_pose(z, landmark, sigma):
    """""
    Sample a robot pose from the landmark model
    Inputs:
        - z: the measurements features (range and bearing of the landmark from the sensor) [r, phi]
        - landmark: the landmark position in the map [m_x, m_y]
        - sigma: the standard deviation of the measurement noise [sigma_r, sigma_phi]
    Outputs:
        - x': the sampled robot pose [x', y', theta']
    """""
    m_x, m_y = landmark[:]
    sigma_r, sigma_phi = sigma[:]

    gamma_hat = np.random.uniform(0, 2*np.pi)
    r_hat = z[0] + np.random.normal(0, sigma_r)
    phi_hat = z[1] + np.random.normal(0, sigma_phi)

    x_ = m_x + r_hat * cos(gamma_hat)
    y_ = m_y + r_hat * sin(gamma_hat)
    theta_ = gamma_hat - np.pi - phi_hat

    return np.array([x_, y_, theta_])


def plot_sampled_poses(robot_pose, z, landmark, sigma, n_samples):
    """""
    Plot sampled poses from the landmark model
    """""
    # plot samples poses
    for i in range(n_samples):
        x_prime = landmark_model_sample_pose(z, landmark, sigma)
        # plot robot pose
        rotated_marker = mpl.markers.MarkerStyle(marker=arrow)
        rotated_marker._transform = rotated_marker.get_transform().rotate_deg(degrees(x_prime[2])-90)
        plt.scatter(x_prime[0], x_prime[1], marker=rotated_marker, s=80, facecolors='none', edgecolors='b')
    
    # plot real pose
    rotated_marker = mpl.markers.MarkerStyle(marker=arrow)
    rotated_marker._transform = rotated_marker.get_transform().rotate_deg(degrees(robot_pose[2])-90)
    plt.scatter(robot_pose[0], robot_pose[1], marker=rotated_marker, s=140, facecolors='none', edgecolors='r')

    plt.xlabel("x-position [m]")
    plt.ylabel("y-position [m]")
    plt.title("Landmark Model Pose Sampling")
    plt.savefig("landmark_model_pose_sampling.pdf")
    plt.show()


def plot_landmarks(landmarks, robot_pose, z, p_z, max_range=6.0, fov=np.pi/4):
    """""
    Plot landmarks, robot pose with sensor FOV, and detected landmarks with associated probability
    """""
    x, y, theta = robot_pose[:]

    start_angle = theta + fov/2
    end_angle = theta - fov/2

    plt.figure()
    ax = plt.gca()
    # plot robot pose
    # find the virtual end point for orientation
    endx = x + 0.5 * cos(theta)
    endy = y + 0.5 * sin(theta)
    plt.plot(x, y, 'or', ms=10)
    plt.plot([x, endx], [y, endy], linewidth = '2', color='r')

    # plot FOV
    # get ray target coordinates
    fov_x_left = x + cos(start_angle) * max_range
    fov_y_left = y + sin(start_angle) * max_range
    fov_x_right = x + cos(end_angle) * max_range
    fov_y_right = y + sin(end_angle) * max_range

    plt.plot([x, fov_x_left], [y, fov_y_left], linewidth = '1', color='b')
    plt.plot([x, fov_x_right], [y, fov_y_right], linewidth = '1', color='b')

    R = max_range
    a, b = 2*R, 2*R
    arc = Arc((x, y), a, b,
                 theta1=degrees(end_angle), theta2=degrees(start_angle), color='b', lw=1.2)
    ax.add_patch(arc)

    # plot landmarks
    for i, lm in enumerate(landmarks):
        plt.plot(lm[0], lm[1], "sk", ms=10, alpha=0.7)

    # plot perceived landmarks position and associated probability (color scale)
    lm_z = np.zeros((len(z), 2))
    for i in range(len(z)):
        # draw endpoint with probability from Likelihood Fields
        lx = x + z[i][0] * cos(z[i][1]+theta)
        ly = y + z[i][0] * sin(z[i][1]+theta)
        lm_z[i, :] = lx, ly
    
    col = np.array(p_z)
    plt.scatter(lm_z[:,0], lm_z[:,1], s=60, c=col, cmap='viridis')
    plt.colorbar()
    plt.savefig(f"landmarks.pdf")
    plt.show()
    plt.close('all')

# ------------------------ Main ------------------------

def main():
    plt.close('all')
    n_samples = 500
    dt = 0.5

    # initial pose
    x = np.array([2., 4., np.pi/6])
    # velocity command (linear and angular) 
    u = np.array([0.8, 0.6])
    # noise variance: [[highlight linear motion uncertainty],[highlight angular motion uncertainty]]
    alpha = [[0.1, 0.1, 0.001, 0.02, 0.05, 0.05],[0.001, 0.001, 0.1, 0.2, 0.05, 0.05]] 
    graph_titles = ["High linear motion uncertainty", "High angular motion uncertainty"]

    # Variables for probabilistic measurement model - landmark model
    # landmarks position in the map
    landmarks = [
                 np.array([5., 2.]),
                 np.array([-2.5, 3.]),
                 np.array([3., 1.5]),
                 np.array([4., -1.]),
                 np.array([-2., -2.])
                 ]
    # sensor parameters
    fov = np.pi/3
    max_range = 6.0
    sigma = np.array([0.3, np.pi/24])

    counter = 0   # used to associate the correct title of the graph to the corresponding set of alpha values used

    print("\n---------- Probabilistic velocity-based motion model ----------\n")
    
    for a in alpha:
        x_prime = np.zeros([n_samples, 3])
        for i in range(n_samples):
            x_prime[i,:] = sample_velocity_motion_model(x, u, a, dt)

        # ---------- Plot x samples ---------- 
        
        mu = np.mean(x_prime, axis=0)
        std_dev = np.std(x_prime, axis=0)  # sarebbe sigma, cambiato per non confonderlo con sigma del punto successivo
        evaluate_sampling_dist(mu[0], std_dev[0], n_samples, np.random.normal, graph_titles[counter])

        # ---------- Sampling the velocity model ---------- 

        rotated_marker = mpl.markers.MarkerStyle(marker=arrow)
        rotated_marker._transform = rotated_marker.get_transform().rotate_deg(degrees(x[2])-90)
        plt.scatter(x[0], x[1], marker=rotated_marker, s=100, facecolors='none', edgecolors='b')

        for x_ in x_prime:
            rotated_marker = mpl.markers.MarkerStyle(marker=arrow)
            rotated_marker._transform = rotated_marker.get_transform().rotate_deg(degrees(x_[2])-90)
            plt.scatter(x_[0], x_[1], marker=rotated_marker, s=40, facecolors='none', edgecolors='r')

        plt.xlabel("x-position [m]")
        plt.ylabel("y-position [m]")
        plt.title(f"velocity motion model sampling ({graph_titles[counter]})")
        plt.savefig(f"velocity_samples_{graph_titles[counter].split()[1]}.pdf")
        plt.show()

        counter += 1

    # ---------- Compute Jacobians ---------- 
    
    print("---------- Jacobians w.r.t. state and command ----------\n")
    Gt, Vt = compute_jacobians_vel_model(x, u, dt)  
    print("Gt:\n", Gt)
    print("\nVt:\n", Vt)


    # ---------- Landmark model (sampling) ---------- 
    
    print("\n---------- Probabilistic measurement model ----------\n")
    x = [0., 0.5, 0]  # robot pose
    z = []
    p = []
    for i in range(len(landmarks)):
        # read sensor measurements (range, bearing)
        z_i = landmark_range_bearing_sensor(x, landmarks[i], sigma=sigma, max_range=max_range, fov=fov)
        
        if z_i is not None: # if landmark is not detected, the measurement is None
            z.append(z_i)
            # compute the probability for each measurement according to the landmark model algorithm
            p_i = landmark_model_prob(z_i, landmarks[i], x, max_range, fov, sigma)
            p.append(p_i)

    print("Probability density value:", np.array(p))
    # Plot landmarks, robot pose with sensor FOV, and detected landmarks with associated probability
    plot_landmarks(landmarks, x, z, p, fov=fov)


    # ---------- Sampling poses from landmark model ---------- 

    n_samples = 1000

    if len(z) == 0:
        print("No landmarks detected!")
        return
    
    # consider only the first landmark detected
    landmark = landmarks[0]
    z = landmark_range_bearing_sensor(x, landmark, sigma)

    # plot landmark
    plt.plot(landmark[0], landmark[1], "sk", ms=10)
    plot_sampled_poses(x, z, landmark, sigma, n_samples)

    plt.close('all')

    # ---------- Compute Jacobian wrt state ----------
    
    print("---------- Jacobian w.r.t. state ----------\n")
    Ht = compute_jacobian_sensor_model(x, landmark)
    print("Ht:\n", Ht)



if __name__ == "__main__":
    main()