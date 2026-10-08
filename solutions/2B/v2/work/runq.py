"""GPU job queue: each line 'ARCH EPOCHS SEED [k=v ...]' -> 5 fold subprocesses, skip done, retry on failure (OOM from shared GPU)."""
import os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.expanduser("~/anaconda3/envs/ml/bin/python")
qf = sys.argv[1]
done_lines = set()
while True:
    lines = [l.split() for l in open(qf) if l.strip() and not l.startswith("#")]
    todo = [l for l in lines if " ".join(l) not in done_lines]
    if not todo:
        break
    l = todo[0]
    arch, ep, seed, extra = l[0], l[1], l[2], l[3:]
    tag = next((e.split("=")[1] for e in extra if e.startswith("tag=")), f"{arch}_e{ep}_s{seed}")
    for k in range(5):
        out = f"{HERE}/out/{tag}_f{k}.npz"
        for attempt in range(8):
            if os.path.exists(out):
                break
            r = subprocess.run([PY, "img_cv.py", arch, ep, seed, str(k)] + extra, cwd=HERE,
                               stdout=open(f"{HERE}/logs/{tag}_f{k}.log", "a"), stderr=subprocess.STDOUT)
            if r.returncode != 0:
                time.sleep(60)
    done_lines.add(" ".join(l))
print("queue done", flush=True)
