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
from human_robot_gym.wrappers.ik_waypoints_delta_wrapper import IKWayPointsDeltaWrapper
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
            update_kwargs["renderer"] = "mjviewer"
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
        use_failsafe_controller = kwargs.get("use_failsafe_controller", True)
        use_waypoints_action = kwargs.get("use_waypoints_action", False)
        n_waypoints = kwargs.get("n_waypoints", 1)
        shield_type = kwargs.get("shield_type", "OFF")

        original_controller_config = deepcopy(kwargs["controller_configs"])
        if use_failsafe_controller:
            pybullet_urdf_file = file_path_completion("models/assets/robots/panda/panda_with_gripper.urdf") 
            robot_config_path = file_path_completion("models/robots/config/panda.json")
            import json
            if shield_type == "CBF":
                failsafe_config_path = file_path_completion("controllers/failsafe_controller/config/cbf_failsafe.json")
            else:
                failsafe_config_path = file_path_completion("controllers/failsafe_controller/config/failsafe.json")
            with open(failsafe_config_path, 'r') as f:
                failsafe_config = json.load(f)
            # Load robot-specific limits
            with open(robot_config_path, 'r') as f:
                robot_config = json.load(f)
            # Merge robot limits into failsafe config
            controller_config = {'body_parts': {'right': {}}}
            controller_config['body_parts']['right'] = merge_configs(failsafe_config['body_parts']['right'], robot_config)
            controller_configs = [controller_config]
        else:
            controller_configs = original_controller_config
            if use_waypoints_action:
                raise NotImplementedError("Waypoint actions require the failsafe controller.")

        update_kwargs = dict(
            robots="Panda",
            robot_base_offset=[0, 0, 0],
            control_freq=kwargs.get("control_freq", 20) / n_waypoints,
            control_sample_time=kwargs.get("model_timestep", 0.002),
            horizon=kwargs.get("max_steps", 400),
            hard_reset=False,
            controller_configs=controller_configs,
            use_failsafe_controller=use_failsafe_controller,
            shield_type=shield_type,
            visualize_failsafe_controller=False,
            visualize_pinocchio=False,
            base_human_pos_offset=[0.0, 0.0, 0.0],
            verbose=False,  # Enable verbose output for debugging
            goal_dist=0.0001,
            human_rand=[0.0, 0.0, 0.0],
            human_animation_names=["SinglePoint/left_right"],
            human_animation_freq=20,
            use_waypoints_action=use_waypoints_action,
            n_waypoints=n_waypoints,
        )
        kwargs.update(update_kwargs)

        # Create robosuite environment directly (don't use GymWrapper to preserve dict obs)
        env = suite.make(
            env_name,
            **kwargs  # pass through any other kwargs
        )

        if use_failsafe_controller:
            pybullet_urdf_file = file_path_completion(
                "models/assets/robots/panda/panda_with_gripper.urdf"
            )
            if use_waypoints_action:
                env = IKWayPointsDeltaWrapper(
                    env=env,
                    urdf_file=pybullet_urdf_file,
                    action_limits=[
                        original_controller_config['body_parts']['right'].get('output_min', [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5]),
                        original_controller_config['body_parts']['right'].get('output_max', [0.05, 0.05, 0.05, 0.5, 0.5, 0.5])
                    ],
                    use_orientation=original_controller_config['body_parts']['right'].get('input_type', "delta")=="delta",
                    n_waypoints=n_waypoints
                )
            else:
                env = IKPositionDeltaWrapper(
                    env=env,
                    urdf_file=pybullet_urdf_file,
                    action_limits=[
                        original_controller_config['body_parts']['right'].get('output_min', [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5]),
                        original_controller_config['body_parts']['right'].get('output_max', [0.05, 0.05, 0.05, 0.5, 0.5, 0.5])
                    ],
                    use_orientation=original_controller_config['body_parts']['right'].get('input_type', "delta")=="delta"
                )

        return env
