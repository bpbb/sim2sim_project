# Quick Status Update

## What Happened

### ❌ First Training Run (23:12-23:13) - TOO SHORT
- Only did **7 PPO updates** (completed in 1 minute!)
- Issue: I specified `--steps 50000` thinking it meant policy updates
- Actually: The script interprets "steps" as environment steps
- Result: 50,000 ÷ (64 envs × 100 horizon) = 7 updates only

### ✅ Second Training Run (23:17) - PROPER
- Now running with **640,000 environment steps = 100 PPO updates**
- This will take ~30-60 minutes (proper training time)
- Process ID: 255741
- CPU usage: 104% (active training)

## Check Training Status

```bash
# See if still running
ps aux | grep "[f]inetune_mujoco.py"

# View log (output is buffered, may take time to appear)
tail -f experiments/go2/4_sim2sim_adaptation/finetune_log.txt

# Check CPU usage
top -p 255741
```

## After Training Completes

The policy will be saved to:
- `experiments/go2/4_sim2sim_adaptation/moderate_adapted.pt`

Then run evaluation:
```bash
bash experiments/go2/4_sim2sim_adaptation/WHEN_DONE.sh
```

## Expected Timeline

- **Start**: 23:17
- **Expected end**: 23:47 - 00:17 (30-60 min from start)
- **Check back at**: ~00:00 to see if done

## Why This Takes Time

- 100 PPO updates × (64 envs × 100 steps) = 640,000 environment steps
- Each step involves: policy forward, MuJoCo physics, reward compute
- Plus: 4 PPO epochs per update for gradient optimization
- Total: ~30-60 minutes on this hardware

## What It's Doing

```
For each of 100 updates:
  1. Collect 100 steps × 64 envs = 6,400 rollout steps
  2. Compute advantages and returns
  3. Run 4 PPO optimization epochs
  4. Update policy weights
  5. Repeat
```

The policy learns to adapt from Isaac Lab dynamics to MuJoCo dynamics!
