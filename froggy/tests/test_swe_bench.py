import os

from froggy.envs.swe_bench import SWEBenchEnv


def test_load_dataset(tmp_path):
    working_dir = str(tmp_path)
    swe_env = SWEBenchEnv(path=working_dir)
    assert swe_env.dataset_id == "princeton-nlp/SWE-bench_Verified"
    # check if the dataset contains features that SWEBenchEnv expects
    assert list(swe_env.ds.features.keys()) == [
        'repo',
        'instance_id',
        'base_commit',
        'patch',  # not required
        'test_patch',
        'problem_statement',
        'hints_text',  # not required
        'created_at',  # not required
        'version',  # not required
        'FAIL_TO_PASS',
        'PASS_TO_PASS',
        'environment_setup_commit',  # not required
    ]


def test_clone_repo(tmp_path):
    working_dir = str(tmp_path)
    swe_env = SWEBenchEnv(path=working_dir)
    task_name = "astropy__astropy-14096"
    row = swe_env.dataset[task_name]
    repo_address = row["repo"]
    local_repo_path = swe_env.clone_repo(repo_address)
    repo_content = os.listdir(local_repo_path)
    assert "astropy" in repo_content


def test_mak_froggyignore(tmp_path):
    working_dir = str(tmp_path)
    swe_env = SWEBenchEnv(path=working_dir)
    task_name = "astropy__astropy-14096"
    row = swe_env.dataset[task_name]
    repo_address = row["repo"]
    local_repo_path = swe_env.clone_repo(repo_address)
    swe_env.make_froggyignore(local_repo_path, include_gitignore=False)
    with open(local_repo_path / ".froggyignore", "r") as f:
        froggyignore = f.read()
    assert froggyignore == "*/tests/\n.froggyignore"


def test_mak_froggyignore_include_gitignore(tmp_path):
    working_dir = str(tmp_path)
    swe_env = SWEBenchEnv(path=working_dir)
    task_name = "astropy__astropy-14096"
    row = swe_env.dataset[task_name]
    repo_address = row["repo"]
    local_repo_path = swe_env.clone_repo(repo_address)
    swe_env.make_froggyignore(local_repo_path)
    with open(local_repo_path / ".froggyignore", "r") as f:
        froggyignore = f.read()
    assert froggyignore.startswith("*/tests/\n.froggyignore")
    assert len(froggyignore.split("\n")) > 2

