import numpy as np
from scipy.optimize import minimize

focalLengthDepth = (774.1458740234375, 774.1458740234375)
max_depth = 10.0
desired_distance = 0.75
frame_width = 640
dia = 0.5
class AdaptiveMPC:
    def __init__(self):
        # MPC weights (tune these)
        self.R = np.diag([0.2,  1.0])   # Control effort weights [linear, angular]
        self.dt = 0.1  # Time step for the model
        # Physical constraints
        self.max_linear_vel = 0.6      # m/s (QBot max ~1.0m/s)
        self.max_angular_vel = 1.5     # rad/s
        self.safe_distance = 1.05       # meters
        self.control_horizon = 2
        self.prediction_horizon = 5   # Number of steps to predict

    def mpc_cost(self, u, x0, V_human_lat, V_human_axial):
        cost = 0.0
        x = x0.copy()
        v, w = u

        for k in range(self.prediction_horizon):
            # 1) reconstruct actual distance
            gehrai = x[1]*max_depth + desired_distance
            fw_m   = (frame_width/2) * gehrai / focalLengthDepth[0]

            # 2) compute a smooth "ideal speed" v_ref:
            #    • zero at very close (<1 m)
            #    • ramps linearly to max at 4 m
            #    • caps at self.max_linear_vel beyond that
            lower_safe    = 0.1
            upper_comfort = 2.75
            v_ref = ((gehrai - lower_safe)
                    / (upper_comfort - lower_safe)
                    ) * self.max_linear_vel
            v_ref = np.clip(v_ref, 0.0, self.max_linear_vel)

            # 3) penalize deviation from v_ref
            q_v = 15.0      # tune this: higher → track v_ref more tightly
            cost += q_v * (v - v_ref)**2

            # 4) propagate your linearized model
            x[0] += ((V_human_lat + w * (gehrai + dia/2)) / fw_m) * self.dt
            x[1] += ((v - V_human_axial)/max_depth) * self.dt

            # 5) keep the exponential error penalties
            cost += 8.0  * (1 - np.exp(-3.0 * x[0]**2))
            cost += 20.0 * (1 - np.exp(-2.0 * x[1]**2))

            # 6) control‐effort penalty on first few steps
            if k < self.control_horizon:
                cost += u.T @ self.R @ u
                cost += 20.0 * (w**2)

        return cost

    
    def get_velocity(self, bbox, frame_width, depth_val, V_human_lat, V_human_axial):   
        if bbox is None:
            return 0.0, 0.0  # Stop if no position available
        
        # Get center of bounding box
        person_x = bbox[0] + bbox[2]/2
        # Calculate error from center of frame
        error_x = (person_x - frame_width/2) / (frame_width/2)
        # Use depth information to convert error to meters
        error_x_meter = (person_x - frame_width/2) * depth_val / focalLengthDepth[0]
        
        distance_factor = 1.0
        if depth_val is not None:
            # Adjust control based on depth (slower when closer)
            if depth_val < 1.5:  # Closer than 2 meter
                distance_factor = 0.5  # Reduce speed
            elif depth_val > 3.0:  # Further than 3 meters
                distance_factor = 1.5  # Increase speed
              
        # Initial guess from proportional controller  
        v_prop = 0.3 * (1 - abs(error_x)) * distance_factor
        w_prop = -(0.8) * error_x  # Angular velocity
        print(f"initial guess: v_prop: {v_prop}, w_prop: {w_prop}")
        u0 = np.array([v_prop, w_prop])
        
        # print(f"v_prop: {v_prop}, w_prop: {w_prop}")
        # Control bounds (v ≥ 0, |ω| ≤ max)
        bounds = [
            (0, self.max_linear_vel),
            (-self.max_angular_vel, self.max_angular_vel)
        ]
        
        depth_error = (depth_val - desired_distance)/max_depth
        x0 = np.array([error_x, depth_error])
        print(f"Initial state: {x0}")
        # Solve MPC optimization
        res = minimize(
            self.mpc_cost, u0,
            args=(x0, V_human_lat, V_human_axial),
            bounds=bounds,
            method='SLSQP',
            options={'maxiter': 50}
        )

        # Extract solution or fallback
        if res.success:
            v_opt, w_opt = res.x
        else:
            v_opt, w_opt = u0  # Fallback to proportional
        
        # Clamp to physical limits
        return (
            np.clip(v_opt, 0, self.max_linear_vel),
            np.clip(w_opt, -self.max_angular_vel, self.max_angular_vel)
        )
