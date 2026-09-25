import numpy as np
from numpy.linalg import inv
from lab04_pkg.utils import normalize_angle

class RobotEKF:
    def __init__(
        self,
        dim_x=1,
        dim_u=1,
        eval_gux=None,
        eval_Gt=None,
        eval_Vt=None,
    ):
        """
        Initializes the extended Kalman filter creating the necessary matrices
        """
        self.mu = np.zeros((dim_x))  # mean state estimate
        self.Sigma = np.eye(dim_x)  # covariance state estimate
        self.Mt = np.eye(dim_u)  # process noise

        self.eval_gux = eval_gux
        self.eval_Gt = eval_Gt
        self.eval_Vt = eval_Vt

        self._I = np.eye(dim_x)  # identity matrix used for computations

    def predict(self, u, sigma_u, g_extra_args=()):
        """
        Update the state prediction using the control input u and compute the relative uncertainty ellipse
        """
        # 1. Estrai dt da g_extra_args
        if isinstance(g_extra_args, (tuple, list)) and len(g_extra_args) > 0:
            dt = g_extra_args[0]
        else:
            dt = 0.05  # default

        # 3. Predizione stato
        self.mu = self.eval_gux(self.mu, u, sigma_u, g_extra_args)

        # 4. args per Gt e Vt: [x, y, theta, v, w, dt]
        args = (*self.mu, *u, dt)  # ← Ora ha 6 elementi!

        # 5. Calcola Jacobiane
        Gt = self.eval_Gt(*args)  
        Vt = self.eval_Vt(*args) 

        # 6. Predizione covarianza
        self.Sigma = Gt @ self.Sigma @ Gt.T + Vt @ self.Mt @ Vt.T

    def update(self, z, eval_hx, eval_Ht, Qt, Ht_args=(), hx_args=(),  residual=np.subtract, **kwargs):
        """Performs the update innovation of the extended Kalman filter.

        Parameters
        ----------

        z : np.array
            measurement for this step.

        lmark : [x, y] list-like
            Landmark location in cartesian coordinates.

        residual : function (z, z2), optional
            Optional function that computes the residual (difference) between
            the two measurement vectors. If you do not provide this, then the
            built in minus operator will be used. You will normally want to use
            the built in unless your residual computation is nonlinear (for
            example, if they are angles)
        """

        # Convert the measurement to a vector if necessary. Needed for the residual computation
        if np.isscalar(z):
            z = np.asarray([z], float)
            Qt = np.atleast_2d(Qt).astype(float)
            Ht = np.atleast_2d(Ht).astype(float)

        # Compute the Kalman gain, you need to evaluate the Jacobian Ht
        Ht = eval_Ht(*Ht_args)
        
        SigmaHT = self.Sigma @ Ht.T
        self.S = Ht @ SigmaHT + Qt
        self.K = SigmaHT @ inv(self.S)

        # Evaluate the expected measurement and compute the residual, then update the state prediction
        z_hat = eval_hx(*hx_args)
        z_hat[1,0] = normalize_angle(z_hat[1,0])

        if np.isscalar(z_hat):
            z_hat = np.asarray([z_hat], float)

        # if the z measurement include an angle update, we need to specify the positional index to normalize the residual
        y = residual(z, z_hat, **kwargs)
        #self.mu = self.mu + self.K @ y
        correction=self.K @ y
        #print(f"z = {z} \n\nz_hat= {z_hat}\n\nsigmaHt={SigmaHT}\n\nHt = {Ht}\n\nsigma = {self.Sigma}")

        # self.mu=self.mu+correction.flatten()  # codice di sergio
        #print(f"K = {self.K} \n\ny= {y}\n\ncorrection={correction}\n\nself.mu = {self.mu}")
        self.mu=self.mu+correction.flatten()

        # Normalizza l'angolo tra -pi e pi
        self.mu[2] = (self.mu[2] + np.pi) % (2 * np.pi) - np.pi

        # P = (I-KH)P(I-KH)' + KRK' is more numerically stable and works for non-optimal K vs the equation
        # P = (I-KH)P usually seen in the literature.
        # Note that I is the identity matrix.
        I_KH = self._I - self.K @ Ht
        self.Sigma = I_KH @ self.Sigma @ I_KH.T + self.K @ Qt @ self.K.T