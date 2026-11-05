#!/usr/bin/env python3
"""
Master Verification & Test Harness for sfm-snr-mfsr.

Validates:
  1. Unit Test: Exact Adjoint Matrix Operator Dot-Product (<Ax, y> == <x, A^T y>).
  2. Direction 1: SR + Denoising Preprocessing for SfM & Bundle Adjustment.
  3. Direction 2: Photogrammetric MVS + MAP-BTV Multi-Frame Super-Resolution & SNR Boost.
  4. C++ Universal Hardware-Agnostic Engine Execution & Device Detection.
"""

import os
import sys
import subprocess
import time

def run_test(name, cmd, cwd=None):
    print(f"\n================================================================================")
    print(f"  RUNNING TEST: {name}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"================================================================================")
    t0 = time.time()
    env = os.environ.copy()
    env['PATH'] = r'C:\msys64\ucrt64\bin;' + env.get('PATH', '')
    res = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    dt = time.time() - t0
    
    if res.stdout:
        print(res.stdout.strip())
    if res.stderr:
        print("STDERR:\n" + res.stderr.strip())
        
    print(f"\n>> Finished {name} in {dt:.2f}s with Exit Code: {res.returncode}")
    if res.returncode != 0:
        print(f"FAILED: {name}")
        return False, res.stdout
    print(f"PASSED: {name}")
    return True, res.stdout

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("Base Directory:", base_dir)
    
    results = {}
    
    # 1. Adjoint Operator Test
    test_adj = os.path.join(base_dir, "python", "test_adjoint.py")
    ok1, out1 = run_test("1. Exact Adjoint Dot-Product Test", [sys.executable, test_adj])
    results["Adjoint Test"] = "PASS" if ok1 else "FAIL"
    
    # 2. Python Reference Pipeline
    py_sfm = os.path.join(base_dir, "python", "sfm_mvs_sr.py")
    ok2, out2 = run_test("2. Python Full End-to-End Pipeline", [sys.executable, py_sfm])
    results["Python Pipeline"] = "PASS" if ok2 else "FAIL"
    
    # 3. C++ Universal Engine
    cpp_exe = os.path.join(base_dir, "sfm_mvs_sr.exe")
    if os.path.exists(cpp_exe):
        ok3, out3 = run_test("3. C++20 Hardware-Agnostic Engine", [cpp_exe], cwd=base_dir)
        results["C++ Engine"] = "PASS" if ok3 else "FAIL"
    else:
        results["C++ Engine"] = "SKIPPED (binary not found)"
        
    print("\n" + "=" * 80)
    print("  TEST HARNESS SUMMARY")
    print("=" * 80)
    for k, v in results.items():
        print(f"  {k:35s}: {v}")
    print("=" * 80)
    
    if all(v == "PASS" for v in results.values()):
        print("\nALL FLOSS TEST SUITES PASSED SUCCESSFULLY!")
        return 0
    else:
        print("\nSOME TESTS FAILED.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
