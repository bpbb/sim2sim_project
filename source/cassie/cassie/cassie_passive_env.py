# cassie_passive_env.py

from __future__ import annotations

import torch
import gymnasium as gym

from .cassie_env import CassieEnv
from .cassie_sim2sim_passive_cfg import CassieSim2SimPassiveEnvCfg

class CassiePassiveEnv(CassieEnv):
    """Cassie environment with passive ankles (10 active joints).
    
    Subclasses CassieEnv to handle the mismatch between:
    - 10-dim action space (Passive Ankles)
    - 12-dim robot DOFs (All joints)
    
    It maps the 10 actions to the 12 joints (filling ankles with 0.0)
    before sending to the simulation.
    """

    cfg: CassieSim2SimPassiveEnvCfg

    def __init__(self, cfg: CassieSim2SimPassiveEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Identify joint indices
        # We need to map 10 actions -> 12 targets
        # Find which indices are not ankles
        joint_names = self._robot.joint_names
        self._active_indices = []
        self._passive_indices = []
        
        for i, name in enumerate(joint_names):
            if "ankle" in name:
                self._passive_indices.append(i)
            else:
                self._active_indices.append(i)
                
        self._active_indices = torch.tensor(self._active_indices, device=self.device, dtype=torch.long)
        
        # Verify sizes
        if len(self._active_indices) != 10:
            print(f"[CassiePassiveEnv] WARNING: Expected 10 active joints, found {len(self._active_indices)}")
            print(f"[CassiePassiveEnv] Active: {[joint_names[i] for i in self._active_indices]}")

        print(f"[CassiePassiveEnv] Passive Joints (Ankles): {[joint_names[i] for i in self._passive_indices]}")

    def _pre_physics_step(self, actions: torch.Tensor):
        """Process actions: Map 10-dim actions to 12-dim targets."""
        self._actions = actions.clone()
        
        # Create full 12-dim vector
        # Apply scaling
        scaled_actions = self.cfg.action_scale * self._actions
        
        # Create full target vector matching default_joint_pos shape (num_envs, 12)
        full_targets = self._robot.data.default_joint_pos.clone()
        
        # Update only active indices
        # We add the scaled action to the default position for active joints
        # actions: (num_envs, 10). active_indices: (10,)
        # clone() is needed if we modify in place, but here we can just add
        
        # full_targets[:, self._active_indices] += scaled_actions
        # Indexing with broadcasting
        full_targets.index_add_(1, self._active_indices, scaled_actions)
        
        self._processed_actions = full_targets

        # Resample commands logic from base class
        self._command_time_left -= self.step_dt
        resample_ids = torch.where(self._command_time_left <= 0.0)[0]
        if len(resample_ids) > 0:
            self._resample_commands(resample_ids)
            self._command_time_left[resample_ids] = self.cfg.command_resampling_time
            
    # Base class _apply_action uses self._processed_actions, so it works automatically.
