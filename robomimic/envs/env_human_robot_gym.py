from robomimic.envs.env_base import EnvBase, EnvType
import robosuite as suite
# from robosuite.wrappers import GymWrapper
import numpy as np
from robomimic.envs.env_robosuite import EnvRobosuite
import robomimic.utils.obs_utils as ObsUtils
from human_robot_gym.utils.mjcf_utils import (
    file_path_completion, 
    merge_configs
)
from human_robot_gym.wrappers.ik_position_delta_wrapper import IKPositionDeltaWrapper
from copy import deepcopy
import robomimic.utils.lang_utils as LangUtils
import robosuite


class EnvHumanRobotGym(EnvRobosuite):
    """A wrapper for Human-Robot Gym environments."""
    def __init__(
        self, 
        env_name, 
        render=False, 
        render_offscreen=False, 
        use_image_obs=False, 
        use_depth_obs=False, 
        lang=None,
        **kwargs,
    ):
        self.use_depth_obs = use_depth_obs

        # robosuite version check
        self._is_v1 = (robosuite.__version__.split(".")[0] == "1")
        if self._is_v1:
            assert (int(robosuite.__version__.split(".")[1]) >= 2), "only support robosuite v0.3 and v1.2+"

        kwargs = deepcopy(kwargs)

        # update kwargs based on passed arguments
        update_kwargs = dict(
            has_renderer=render,
            has_offscreen_renderer=(render_offscreen or use_image_obs),
            ignore_done=True,
            use_object_obs=True,
            use_camera_obs=use_image_obs,
            camera_depths=use_depth_obs,
        )
        if render and self.is_v15_or_higher:
            update_kwargs["renderer"] = "mujoco"
        kwargs.update(update_kwargs)

        if self._is_v1:
            if kwargs["has_offscreen_renderer"]:
                # ensure that we select the correct GPU device for rendering by testing for EGL rendering
                # NOTE: this package should be installed from this link (https://github.com/StanfordVL/egl_probe)
                import egl_probe
                valid_gpu_devices = egl_probe.get_available_devices()
                if len(valid_gpu_devices) > 0:
                    kwargs["render_gpu_device_id"] = valid_gpu_devices[0]
        else:
            # make sure gripper visualization is turned off (we almost always want this for learning)
            kwargs["gripper_visualization"] = False
            del kwargs["camera_depths"]
            kwargs["camera_depth"] = use_depth_obs # rename kwarg

        self._env_name = env_name

        self._init_kwargs = deepcopy(kwargs)
        self.env = self._create_safe_env(env_name, **kwargs)
        self.lang = lang
        self._lang_emb = LangUtils.get_lang_emb(self.lang)

    def _create_safe_env(self, env_name, **kwargs):
        # Your exact safety setup code here
        # Setup controller configuration (same as working demo)
        failsafe_config_path = file_path_completion(
            "controllers/failsafe_controller/config/failsafe.json"
        )
        robot_config_path = file_path_completion("models/robots/config/panda.json")

        # Load the failsafe controller config from file
        import json
        with open(failsafe_config_path, 'r') as f:
            failsafe_config = json.load(f)

        # Load robot-specific limits
        with open(robot_config_path, 'r') as f:
            robot_config = json.load(f)

        # Merge robot limits into failsafe config
        controller_config = {'body_parts': {'right': {}}}
        controller_config['body_parts']['right'] = merge_configs(failsafe_config['body_parts']['right'], robot_config)
        controller_configs = [controller_config]

        # Create robosuite environment directly (don't use GymWrapper to preserve dict obs)
        env = suite.make(
            env_name,
            robots="Panda",
            robot_base_offset=[0, 0, 0],
            use_camera_obs=False,  # do not use pixel observations
            has_offscreen_renderer=False,  # not needed since not using pixel obs
            has_renderer=True,  # make sure we can render to the screen
            render_camera=None,
            renderer="mjviewer",
            render_collision_mesh=False,
            reward_shaping=True,  # use dense rewards
            control_freq=5,  # control should happen fast enough
            horizon=kwargs.get("max_steps", 400),
            hard_reset=False,
            controller_configs=controller_configs,
            shield_type="SSM",
            visualize_failsafe_controller=True,  # Enable failsafe viz
            visualize_pinocchio=False,
            base_human_pos_offset=[0.0, 0.0, 0.0],
            verbose=True,  # Enable verbose output for debugging
            goal_dist=0.0001,
            human_rand=[0.0, 0.0, 0.0],
            human_animation_names=["SinglePoint/left_right"],
            human_animation_freq=20
        )

        pybullet_urdf_file = file_path_completion(
            "models/assets/robots/panda/panda_with_gripper.urdf"
        )
        env = IKPositionDeltaWrapper(
            env=env,
            urdf_file=pybullet_urdf_file,
            action_limits=[
                kwargs['controller_configs']['body_parts']['right'].get('output_min', [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5]),
                kwargs['controller_configs']['body_parts']['right'].get('output_max', [0.05, 0.05, 0.05, 0.5, 0.5, 0.5])
            ],
            use_orientation=kwargs['controller_configs']['body_parts']['right'].get('input_type', "delta")=="delta"
        )

        return env

    # def step(self, action):
    #     """Step in the environment with an action."""
    #     return self.env.step(action)

    # def reset(self):
    #     """Reset environment."""
    #     # obs_dict = self.env.reset()
    #     # return self.get_observation(obs_dict)
    #     return self.env.reset()

    # def reset_to(self, state):
    #     """Reset to a specific simulator state."""
    #     should_ret = False
        
    #     if "model" in state:
    #         # Handle model reset if needed
    #         self.reset()
    #         try:
    #             xml = self.env.edit_model_xml(state["model"])
    #             self.env.reset_from_xml_string(xml)
    #             self.env.sim.reset()
    #         except AttributeError:
    #             # Fallback if methods not available
    #             pass
                
    #     if "states" in state:
    #         self.env.sim.set_state_from_flattened(state["states"])
    #         self.env.sim.forward()
    #         should_ret = True
            
    #     if "goal" in state:
    #         self.set_goal(**state["goal"])
            
    #     if should_ret:
    #         return self.env._get_observations(force_update=True)  # self.get_observation()
    #     return None

    # def render(self, mode="human", height=None, width=None, camera_name=None):
    #     """Render environment."""
    #     return self.env.render(mode=mode, height=height, width=width, camera_name=camera_name)

    # def get_observation(self, di=None):
    #     """Get environment observation."""
    #     if di is None:
    #         di = self.env._get_observations(force_update=True)
        
    #     ret = {}
    #     # Add object state
    #     if "object-state" in di:
    #         ret["object"] = np.array(di["object-state"])
        
    #     # Add robot proprioception 
    #     if "robot0_proprio-state" in di:
    #         ret["robot0_proprio"] = np.array(di["robot0_proprio-state"])
        
    #     # Add other standard robosuite observations
    #     for k in di:
    #         if k not in ["object-state", "robot0_proprio-state"] and not k.endswith("_image"):
    #             ret[k] = np.array(di[k])
                
    #     return ret

    # def get_state(self):
    #     """Get environment simulator state."""
    #     xml = self.env.sim.model.get_xml()
    #     state = np.array(self.env.sim.get_state().flatten())
    #     return dict(model=xml, states=state)

    # def get_reward(self):
    #     """Get current reward."""
    #     return self.env.reward()

    # def get_goal(self):
    #     """Get goal observation."""
    #     try:
    #         goal_obs = self.env._get_goal()
    #         return self.get_observation(goal_obs)
    #     except AttributeError:
    #         return {}

    # def set_goal(self, **kwargs):
    #     """Set goal observation."""
    #     try:
    #         return self.env.set_goal(**kwargs)
    #     except AttributeError:
    #         pass

    # def is_done(self):
    #     """Check if the task is done."""
    #     return self.env.done

    # def is_success(self):
    #     """Check if the task is successful."""
    #     succ = self.env._check_success()
    #     if isinstance(succ, dict):
    #         return succ
    #     return {"task": succ}

    # @property
    # def action_dimension(self):
    #     """Returns dimension of actions."""
    #     return self.env.action_spec[0].shape[0]

    # @property
    # def name(self):
    #     """Returns name of environment."""
    #     return self._env_name

    # @property
    # def type(self):
    #     """Returns environment type."""
    #     return EnvType.HUMAN_ROBOT_GYM_TYPE

    # def serialize(self):
    #     """Save all information needed to re-instantiate this environment."""
    #     return dict(
    #         env_name=self.name,
    #         type=self.type,
    #     )

    # @classmethod
    # def create_for_data_processing(
    #     cls,
    #     camera_names,
    #     camera_height,
    #     camera_width,
    #     reward_shaping,
    #     render=None,
    #     render_offscreen=None,
    #     use_image_obs=None,
    #     use_depth_obs=None,
    #     **kwargs,
    # ):
    #     """Create environment for processing datasets."""
    #     has_camera = (len(camera_names) > 0)
    #     return cls(
    #         render=(False if render is None else render),
    #         render_offscreen=(has_camera if render_offscreen is None else render_offscreen),
    #         use_image_obs=(has_camera if use_image_obs is None else use_image_obs),
    #         use_depth_obs=(use_depth_obs or False),
    #         **kwargs,
    #     )

    # @property
    # def rollout_exceptions(self):
    #     """Return tuple of exceptions to except when doing rollouts."""
    #     # Import mujoco exceptions if available
    #     try:
    #         import mujoco_py
    #         return (mujoco_py.builder.MujocoException,)
    #     except ImportError:
    #         return ()

    # @property
    # def base_env(self):
    #     """Grabs base simulation environment."""
    #     return self.env
