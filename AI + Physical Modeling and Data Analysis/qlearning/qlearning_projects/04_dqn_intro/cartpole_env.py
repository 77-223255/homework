"""
CartPole Environment Wrapper
============================

A simple wrapper around the CartPole environment.

CartPole is a classic control problem where you balance a pole
on a moving cart. The state is continuous (4 real numbers).

Author: Educational RL Project
"""

import numpy as np

# Note: We'll implement CartPole manually to avoid gym dependency
# But show how it would work with gym


class CartPole:
    """
    Simple CartPole implementation.
    
    State: [cart_pos, cart_vel, pole_angle, pole_vel]
    Action: 0 = push left, 1 = push right
    
    Goal: Keep pole upright for as long as possible.
    """
    
    # Physics constants
    GRAVITY = 9.8
    MASSCART = 1.0
    MASSPOLE = 0.1
    TOTAL_MASS = MASSPOLE + MASSCART
    LENGTH = 0.5  # Half the pole's length
    POLEMASS_LENGTH = MASSPOLE * LENGTH
    FORCE_MAG = 10.0
    TAU = 0.02  # Seconds between state updates
    
    # Angle limits
    THETA_THRESHOLD = 12 * 2 * np.pi / 360  # 12 degrees
    X_THRESHOLD = 2.4
    
    def __init__(self):
        """Initialize CartPole environment."""
        self.state = None
        self.steps_beyond_done = None
        self.reset()
    
    def reset(self):
        """
        Reset the environment.
        
        Returns:
            state: Initial state (randomized slightly)
        """
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return self.state.copy()
    
    def step(self, action):
        """
        Take an action.
        
        Args:
            action: 0 (left) or 1 (right)
            
        Returns:
            next_state: New state
            reward: Reward received
            done: Whether episode ended
        """
        x, x_dot, theta, theta_dot = self.state
        force = self.FORCE_MAG if action == 1 else -self.FORCE_MAG
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        
        # Physics equations
        temp = (force + self.POLEMASS_LENGTH * theta_dot ** 2 * sintheta) / self.TOTAL_MASS
        thetaacc = (self.GRAVITY * sintheta + costheta * temp) / (
            self.LENGTH * (4.0/3.0 - self.MASSPOLE * costheta ** 2 / self.TOTAL_MASS)
        )
        xacc = temp - self.POLEMASS_LENGTH * thetaacc * costheta / self.TOTAL_MASS
        
        # Update state
        x = x + self.TAU * x_dot
        x_dot = x_dot + self.TAU * xacc
        theta = theta + self.TAU * theta_dot
        theta_dot = theta_dot + self.TAU * thetaacc
        
        self.state = np.array([x, x_dot, theta, theta_dot])
        
        # Check if done
        done = (
            x < -self.X_THRESHOLD or
            x > self.X_THRESHOLD or
            theta < -self.THETA_THRESHOLD or
            theta > self.THETA_THRESHOLD
        )
        
        # Reward
        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            # Already done, shouldn't happen
            reward = 0.0
        
        return self.state.copy(), reward, done
    
    @property
    def observation_space(self):
        """Return observation space info."""
        return type('Space', (), {
            'shape': (4,),
            'low': np.array([-self.X_THRESHOLD*2, -np.inf, -self.THETA_THRESHOLD*2, -np.inf]),
            'high': np.array([self.X_THRESHOLD*2, np.inf, self.THETA_THRESHOLD*2, np.inf])
        })()
    
    @property
    def action_space(self):
        """Return action space info."""
        return type('Space', (), {
            'n': 2
        })()
    
    def render(self):
        """Print current state."""
        x, x_dot, theta, theta_dot = self.state
        angle_deg = theta * 180 / np.pi
        print(f"  Cart: pos={x:6.2f}, vel={x_dot:6.2f}")
        print(f"  Pole: angle={angle_deg:6.2f}°, vel={theta_dot:6.2f}")
    
    def close(self):
        """Close environment."""
        pass


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def test_environment():
    """Test CartPole environment."""
    print("=" * 50)
    print("   CARTPOLE ENVIRONMENT TEST")
    print("=" * 50)
    
    env = CartPole()
    
    print("\nInitial state:")
    state = env.reset()
    env.render()
    
    print("\nTaking random actions...")
    total_reward = 0
    
    for step in range(100):
        action = np.random.randint(2)
        action_name = "RIGHT" if action == 1 else "LEFT"
        
        state, reward, done = env.step(action)
        total_reward += reward
        
        if step % 10 == 0:
            print(f"\nStep {step}: Action={action_name}")
            env.render()
        
        if done:
            print(f"\n💀 Pole fell after {step + 1} steps!")
            break
    
    print(f"\nTotal reward: {total_reward}")


if __name__ == '__main__':
    test_environment()