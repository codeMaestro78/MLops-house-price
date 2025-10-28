import subprocess
import sys

def test_train_runs():
    res=subprocess.run([sys.executable,"-m", "src.train", "--experiment-name", "ci-test-experiment"], check=False)
    assert res.returncode==0


# This test runs training script briefly and checks it exits successfully

