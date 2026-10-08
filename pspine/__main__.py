"""`python -m pspine` -- run the benchmark the README quotes."""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(prog="pspine")
    ap.add_argument("cmd", choices=["bench", "fidelity", "space", "multitable"], nargs="?", default="bench")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--acqs", default="ei,logei,ucb,pi")
    a = ap.parse_args()
    if a.cmd == "space":
        sys.path.insert(0, HERE)
        from pspine import data
        for name in data.TABLES:
            space, codes, y, meta = data.load(name)
            print("== %s: %s" % (name, space))
            for k in ("rows", "grid", "cells_observed", "coverage", "verified",
                      "sha256", "target"):
                if k in meta:
                    print("   %-16s %s" % (k, meta[k]))
            print("   %-16s [%.4g, %.4g]" % ("response range", min(y), max(y)))
        return
    script = os.path.join(HERE, "bench", {
        "bench": "run_campaign.py",
        "fidelity": "gp_fidelity.py",
        "multitable": "run_multitable.py",
    }[a.cmd])
    cmd = [sys.executable, script]
    if a.cmd == "bench":
        cmd += ["--seeds", str(a.seeds), "--budget", str(a.budget), "--acqs", a.acqs]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
