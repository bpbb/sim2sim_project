# scripts/verify_cassie_mujoco.py
import mujoco
import mujoco.viewer
import os

# Path to menagerie model
model_path = os.path.join(
    os.path.dirname(__file__), "..",
    "source/cassie.assets/cassie.assets/data/robots/cassie/mujoco/cassie.xml"
)

model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Print model info
print(f"Model: {model.names}")
print(f"nq (generalized coords): {model.nq}")
print(f"nv (generalized velocities): {model.nv}")
print(f"nu (actuators): {model.nu}")
print(f"nbody: {model.nbody}")

print("\n--- Actuators ---")
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"  [{i}] {name}")

print("\n--- Joints ---")
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    jnt_type = model.jnt_type[i]
    type_names = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
    print(f"  [{i}] {name} ({type_names.get(jnt_type, '?')})")

# Launch viewer
mujoco.viewer.launch(model, data)