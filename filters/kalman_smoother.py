import numpy  as np
from filterpy.kalman import KalmanFilter

class KalmanSmoother:
    def __init__(self, num_joints=33, dt=1/30):
        self.filters = {jid: self._create_filter(dt) for jid in range(num_joints)}

    
    def _create_filter(self, dt):
        kf = KalmanFilter(dim_x=6, dim_z=3)
        kf.F = np.array([[1,0,0,dt,0,0],
                         [0,1,0,0,dt,0],
                         [0,0,1,0,0,dt],
                         [0,0,0,1,0,0],
                         [0,0,0,0,1,0],
                         [0,0,0,0,0,1]])
        kf.H = np.array([[1,0,0,0,0,0],
                         [0,1,0,0,0,0],
                         [0,0,1,0,0,0]])
        
        kf.P *= 0.1
        kf.R *= 0.01
        kf.Q *= 0.01
        kf.x[:3] = 0
        return kf
    
    def update(self, joint_id, X, Y, Z):
        kf = self.filters[joint_id]
        if np.all(kf.x[:3]==0):
            kf.x[:3] = np.array([[X], [Y], [Z]])
        kf.predict()
        kf.update(np.array([X,Y,Z]))
        return kf.x[3].flatten()