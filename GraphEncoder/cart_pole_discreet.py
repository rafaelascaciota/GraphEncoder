"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
Possible to change information on the env
"""
import os
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

class CartPoleEnv(gym.Env):
    """
    Description: A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The
    pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's
    velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson

    Observation:
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -180 deg       180 deg
        3	Pole Velocity At Tip      -Inf            Inf

    Actions:
        Type: Discrete(3)
        Num	Action
        0	Push cart to the left
        1   No force
        2	Push cart to the right

        Note: The amount the velocity that is reduced or increased is not fixed; it depends on the angle the pole is
        pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the
        cart underneath it

    Reward:
        Reward is based on the difference between the current state and the target

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
        Can be set to any angle between -180 to 180 degrees using the reset() function

    Episode Termination:
        Pole Angle is more than 12 degrees --- no longer applicable
        Cart Position is more than 2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'

        # Angle at which to fail the episode
        self.theta_threshold_radians = 45 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        x_coefficient = 1.
        theta_coefficient = 1.
        self.reference = (0., 0., 0., 0.)
        self.max_deviation = np.array([self.x_threshold, 1, self.theta_threshold_radians, 1]) \
                             - np.array(self.reference)
        self.Q = np.array([
            [x_coefficient, 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., theta_coefficient, 0.],
            [0., 0., 0., 0.]
        ])
        self.max_reward = x_coefficient + theta_coefficient
        self.counter = 0
        self.counter_max = 200

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        state = self.state
        self.counter += 1
        x, x_dot, theta, theta_dot = state
        """
        while theta < -np.pi or theta > np.pi:
            if theta > np.pi:
                theta = theta - 2 * np.pi
            if theta < -np.pi:
                theta = 2 * np.pi + theta
        """
        # force = self.force_mag if action == 1 else -self.force_mag
        if action == 2:
            force = self.force_mag
        elif action == 0:
            force = -self.force_mag
        else:
            force = 0.
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
        theta_fixed = np.arcsin(np.sin(theta))
        if np.cos(theta) < 0:
            theta = - np.pi - theta_fixed if theta_fixed < 0 else np.pi - theta_fixed
        self.state = (x, x_dot, theta, theta_dot)
        done = x < -self.x_threshold \
               or x > self.x_threshold \
               or theta < -self.theta_threshold_radians \
               or theta > self.theta_threshold_radians \
               or self.counter >= self.counter_max
        done = bool(done)

        if not done:
            # reward = 1.0
            state_difference = np.asarray(self.state) - np.asarray(self.reference)
            state_difference = np.divide(state_difference, self.max_deviation)
            reward = self.max_reward - 1 * np.matmul(np.matmul(state_difference, self.Q),
                                                     np.transpose(state_difference))
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            # reward = 1.0
            state_difference = np.asarray(self.state) - np.asarray(self.reference)
            state_difference = np.divide(state_difference, self.max_deviation)
            reward = self.max_reward - 1 * np.matmul(np.matmul(state_difference, self.Q),
                                                     np.transpose(state_difference)) - \
                                                    (self.counter_max - self.counter) * 0 - self.counter_max
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = True' -- any further steps are undefined "
                    "behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self, initial_angle=None, duration=None):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        if initial_angle is not None:
            self.state[2] = initial_angle * np.pi / 180
        if duration is not None:
            self.counter_max = duration
        else:
            self.counter_max = 200
        self.steps_beyond_done = None
        self.counter = 0
        return np.array(self.state)

    def render(self, mode='human', color_type=None, show=True):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            # from gym.envs.classic_control import rendering
            import background_rendering as rendering
            self.viewer = rendering.Viewer(screen_width, screen_height, show=show)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None: return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        # cartx = screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array', color_type=color_type)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

def down_scale(input_arr, scale, normalized=False):
    normalized_val = 255. if normalized is True else 1.
    if len(input_arr.shape) == 3:
        (input_size_x, input_size_y, num_channels) = input_arr.shape
    else:
        (input_size_x, input_size_y) = input_arr.shape
        num_channels = 1
    assert input_size_x % scale == 0, "Scaling issue: Output X is not an integer"
    assert input_size_y % scale == 0, "Scaling issue: Output Y is not an integer"
    output_size_x = input_size_x // scale
    output_size_y = input_size_y // scale
    return input_arr.reshape((output_size_x, scale,
                              output_size_y, scale, num_channels)).mean(3).mean(1) / normalized_val

# color_type = {grayscale, black_and_white, None}
if __name__ == "__main__":
    import cv2

    env = CartPoleEnv()
    num_actions = env.action_space.n
    rewards = 0
    steps = 0
    done = False
    env.reset(initial_angle=0.)
    all_folders = os.listdir('./data')

    os.chdir('./data')
    retval = os.getcwd()
    all_folders.sort()
    latest = all_folders[-1].replace('S', '')
    new = int(latest) + 1
    mydir = 'S'+str(new)
    createFolder(mydir)
    os.chdir(mydir)
    while not done:
        temp = env.render(mode='rgb_array', show=False)
        # temp = down_scale(input_arr=temp, scale=20)

        cv2.imwrite('Frame' + str(steps) + '.png', temp)
        action = np.random.randint(num_actions)
        observation, reward, done, _ = env.step(action)
        steps += 1
        rewards += reward
    print("Testing steps: {} rewards {}: ".format(steps, rewards))

    env.close()
