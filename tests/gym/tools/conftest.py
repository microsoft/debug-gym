import pytest


@pytest.fixture
def setup_test_repo():
    def _setup_test_repo(base_dir):
        """Setup a repo with 2 dummy files, 1 fail test, and 1 pass test"""
        working_dir = base_dir / "tests_pdb"
        working_dir.mkdir()
        with (working_dir / "test_pass.py").open("w") as f:
            f.write("def test_pass():\n    assert True")
        with (working_dir / "test_fail.py").open("w") as f:
            f.write("def test_fail():\n    assert False")
        dummy_files = ["file1.py", "file2.py"]
        for dummy_file in dummy_files:
            with (working_dir / dummy_file).open("w") as f:
                [f.write(f"print({i})\n") for i in range(40)]
        return working_dir

    return _setup_test_repo
