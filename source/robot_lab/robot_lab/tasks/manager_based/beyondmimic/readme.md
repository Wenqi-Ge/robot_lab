


`robot_lab/source/robot_lab/robot_lab/tasks/manager_based/beyondmimic/config/g1/__init__.py`

```python
gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-Unitree-G1-v0", # environment's unique identifier
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeG1BeyondMimicFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1BeyondMimicFlatPPORunnerCfg",
    },
)

```


train

```bash
python scripts/reinforcement_learning/rsl_rl/train.py --task=RobotLab-Isaac-BeyondMimic-Flat-Unitree-G1-v0 --headless
```


