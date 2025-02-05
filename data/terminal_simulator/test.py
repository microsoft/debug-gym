from code import run_terminal_simulator

from test_part_1 import *
from test_part_2 import *


def check(candidate):

    def test_all(terminal):
        all_test = [
            test_mkdir,
            test_rmdir,
            test_rmdir_non_empty,
            test_cd,
            test_cd_root,
            test_cd_parent,
            test_list_empty,
            test_list_non_empty,
            test_create_file,
            test_mkdir_rmdir,
            test_create_file_cd_list,
            test_pwd,
            test_ls,
            test_nested_directory_creation,
            test_mkdir_existing_file,
            test_rmdir_nonexistent,
            test_mkdir_nested_nonexistent,
            test_deeply_nested_directory_creation,
            test_create_directory_inside_file,
            test_overwrite_directory_with_file,
            test_create_directory_with_invalid_path,
            test_cd_into_file,
            test_remove_non_empty_directory,
            test_cd_non_existent_parent,
            test_list_non_existent_directory,
            test_remove_directory_with_subdirectories,
            test_cd_parent_from_root,
        ]
        for test in all_test:
            terminal = candidate()
            test(terminal)

    return test_all(candidate)


check(run_terminal_simulator.my_fake_terminal)
