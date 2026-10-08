"""`python -m pspine` -- run the benchmark the README quotes."""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(prog="pspine")
    ap.add_argument("cmd", choices=["bench", "fidelity", "space"], nargs="?", default="bench")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--acqs", default="ei,logei,ucb,pi")
    a = ap.parse_args()
    if a.cmd == "space":
        sys.path.insert(0, HERE)
        from pspine import data
        space, codes, y, meta = data.buchwald()
        print(space)
        for k, v in meta.items():
            print("  %-16s %s" % (k, v))
        return
    script = os.path.join(HERE, "bench",
                          "run_campaign.py" if a.cmd == "bench" else "gp_fidelity.py")
    cmd = [sys.executable, script]
    if a.cmd == "bench":
        cmd += ["--seeds", str(a.seeds), "--budget", str(a.budget), "--acqs", a.acqs]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
