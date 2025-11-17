"""Small wrapper around `filterpy` KalmanFilter to smooth joint 3D positions.

Each joint is assigned a separate 6-state Kalman filter (position +
velocity). The wrapper focuses on creating filters with sensible
defaults and offering a single `update` method that returns the
smoothed (X, Y, Z) coordinates.
"""

import numpy as np
import logging
from filterpy.kalman import KalmanFilter

logger = logging.getLogger(__name__)


class KalmanSmoother:
    def __init__(self, num_joints=33, dt=1/30):
        # Store time step and create a Kalman filter per expected joint id.
        self.dt = dt
        self.filters = {jid: self._create_filter(dt) for jid in range(num_joints)}

    def _init_filter(self, jid):
        """Lazily create and register a Kalman filter for a joint id.

        This is used when an unexpected joint ID is encountered at runtime.
        """
        self.filters[jid] = self._create_filter(self.dt)

    def _create_filter(self, dt):
        """Create and configure a Kalman filter for 3D position smoothing.

        The state vector layout is [x, y, z, vx, vy, vz]. Measurement is
        only position (x,y,z) so H maps state -> position.
        """
        try:
            kf = KalmanFilter(dim_x=6, dim_z=3)
            # State transition: position updated from velocity using dt
            kf.F = np.array([[1, 0, 0, dt, 0, 0],
                             [0, 1, 0, 0, dt, 0],
                             [0, 0, 1, 0, 0, dt],
                             [0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 0, 1]])
            # Measurement matrix: we measure x, y, z directly
            kf.H = np.array([[1, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0]])

            # Tunable covariances — small values give smoother outputs
            kf.P *= 0.1
            kf.R *= 0.01
            kf.Q *= 0.01

            # Initialize position portion of state to zero; velocities stay zero
            kf.x[:3] = 0
            return kf
        except Exception:
            logger.exception("Error creating Kalman filter:")
            return None

    def update(self, jid, X, Y, Z):
        """Update (or initialize) the Kalman filter for `jid` and return
        the smoothed position.

        Returns the input (X, Y, Z) on failure to preserve original data.
        """
        try:
            if jid not in self.filters:
                # Lazily create a filter if a joint id outside the initial
                # range is encountered.
                self._init_filter(jid)

            kf = self.filters[jid]
            # If the filter has not been initialized with a position yet,
            # seed it with the current measurement.
            if np.all(kf.x[:3] == 0):
                kf.x[:3] = np.array([[X], [Y], [Z]])

            # Predict + update cycle
            kf.predict()
            kf.update(np.array([X, Y, Z]))

            X_s, Y_s, Z_s = kf.x[:3].flatten()
            return X_s, Y_s, Z_s

        except Exception:
            logger.exception(f"Error in KalmanSmoother update for joint {jid}:")
            return X, Y, Z
