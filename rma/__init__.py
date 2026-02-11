"""RMA (Rapid Motor Adaptation) module for sim-to-sim transfer.

This module implements RMA for robust policy transfer between simulators.
The key idea is to train a policy with privileged environment information,
then learn an adaptation module that estimates this information from
observation history during deployment.

Components:
- rma_policy.py: RMA policy architecture (encoder + adaptation module)
- rma_env.py: Environment wrapper for privileged observations
- train_rma.py: Two-phase training script
"""

from .rma_policy import (
    EnvironmentEncoder,
    AdaptationModule,
    RMAPolicy,
    RMAPolicyCfg,
)

__all__ = [
    "EnvironmentEncoder",
    "AdaptationModule",
    "RMAPolicy",
    "RMAPolicyCfg",
]
