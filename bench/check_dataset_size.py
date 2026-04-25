import sys
sys.path.insert(0, ".")
import jsb_dataset  # noqa: F401 - registers jsb_* datasets
from constrained_diffusion.eval.dllm.dataset import load_dataset

for name in ["jsb_medium", "jsb_hard", "jsonschema"]:
    try:
        ds = load_dataset(name)
        instances = sorted(ds, key=lambda x: x.instance_id())
        print(f"{name}: {len(instances)} instances")
    except Exception as e:
        print(f"{name}: ERROR - {e}")
