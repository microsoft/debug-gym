import unittest
from unittest.mock import patch, MagicMock, call
from froggy.tools.pdb import PDBTool
from froggy.utils import TimeoutException

class TestPDBTool(unittest.TestCase):
    def setUp(self):
        self.pdb_tool = PDBTool()
        self.pdb_tool.environment = MagicMock()
        self.pdb_tool.environment.working_dir = "/home/user/project"
        self.pdb_tool.master = 1  # Mock the master attribute

    @patch('os.write')
    @patch('os.read', return_value=b'/home/user/project/constants.py(6)<module>()\n-> ACTION_TO_INDEX = {\n(Pdb)')
    def test_get_current_frame_file(self, mock_os_read, mock_os_write):
        self.pdb_tool.has_pseudo_terminal = MagicMock(return_value=True)
        self.pdb_tool.read_pseudo_terminal_output = MagicMock(return_value=(
            "> /home/user/project/constants.py(6)<module>()\n"
            "-> ACTION_TO_INDEX = {\n(Pdb)"
        ))

        # Call the method
        self.pdb_tool.get_current_frame_file()

        # Assertions
        mock_os_write.assert_called_once_with(self.pdb_tool.master, b"where\n")
        self.assertEqual(self.pdb_tool.current_frame_file, "constants.py")

    @patch('os.write')
    @patch('os.read', return_value=b'/home/user/project/constants.py(6)<module>()\n-> ACTION_TO_INDEX = {\n(Pdb)')
    def test_get_current_frame_file_no_pseudo_terminal(self, mock_os_read, mock_os_write):
        self.pdb_tool.has_pseudo_terminal = MagicMock(return_value=False)
        self.pdb_tool.start_pseudo_terminal = MagicMock()
        self.pdb_tool.read_pseudo_terminal_output = MagicMock(return_value=(
            "> /home/user/project/constants.py(6)<module>()\n"
            "-> ACTION_TO_INDEX = {\n"
        ))

        # Call the method
        self.pdb_tool.get_current_frame_file()

        # Assertions
        self.pdb_tool.start_pseudo_terminal.assert_called_once()
        mock_os_write.assert_called_once_with(self.pdb_tool.master, b"where\n")
        self.assertEqual(self.pdb_tool.current_frame_file, "constants.py")

    @patch('os.write')
    @patch('os.read', return_value=b'/other/path/constants.py(6)<module>()\n-> ACTION_TO_INDEX = {\n(Pdb)')
    def test_get_current_frame_file_no_sep_in_output(self, mock_os_read, mock_os_write):
        self.pdb_tool.has_pseudo_terminal = MagicMock(return_value=True)
        self.pdb_tool.read_pseudo_terminal_output = MagicMock(return_value=(
            "/other/path/constants.py(6)<module>()\n"
            "-> ACTION_TO_INDEX = {\n"
        ))

        # Call the method
        self.pdb_tool.get_current_frame_file()

        # Assertions
        mock_os_write.assert_called_once_with(self.pdb_tool.master, b"where\n")
        self.assertIsNone(self.pdb_tool.current_frame_file)

    @patch('subprocess.Popen')
    @patch('os.close')
    @patch('os.read', return_value=b'(Pdb)')
    def test_start_pseudo_terminal(self, mock_os_read, mock_os_close, mock_popen):
        mock_popen.return_value = MagicMock()
        self.pdb_tool.environment.entrypoint = ["python", "script.py"]
        self.pdb_tool.environment.working_dir = "/home/user/project"

        # Call the method
        initial_output = self.pdb_tool.start_pseudo_terminal()

        # Assertions
        self.assertIn("(Pdb)", initial_output)
        mock_popen.assert_called_once_with(
            ["python", "-m", "pdb", "script.py"],
            env=unittest.mock.ANY,
            cwd="/home/user/project",
            stdin=unittest.mock.ANY,
            stdout=unittest.mock.ANY,
            stderr=unittest.mock.ANY,
            text=True,
            close_fds=True,
        )
        #TODO mock_os_close.assert_called_once_with(self.pdb_tool.master)

    @patch('os.write')
    @patch('os.read', return_value=b'(Pdb)')
    def test_interact_with_pseudo_terminal(self, mock_os_read, mock_os_write):
        self.pdb_tool.read_pseudo_terminal_output = MagicMock(return_value="(Pdb)")

        # Call the method
        output = self.pdb_tool.interact_with_pseudo_terminal("list")

        # Assertions
        mock_os_write.assert_called_once_with(self.pdb_tool.master, b"list\n")
        self.assertEqual(output, "(Pdb)")

    @patch('os.write')
    @patch('os.read', side_effect=[b'(Pdb)', b'The program finished and will be restarted\n(Pdb)'])
    def test_interact_with_pseudo_terminal_restart(self, mock_os_read, mock_os_write):
        self.pdb_tool.read_pseudo_terminal_output = MagicMock(side_effect=[
            "(Pdb)",
            "The program finished and will be restarted\n(Pdb)"
        ])
        self.pdb_tool.start_pseudo_terminal = MagicMock(return_value="(Pdb)")

        # Call the method
        output = self.pdb_tool.interact_with_pseudo_terminal("quit")

        # Assertions
        # TODO mock_os_write.assert_has_calls([call(self.pdb_tool.master, b"quit\n")])
        # TODO self.assertIn("The program finished and will be restarted", output)

if __name__ == '__main__':
    unittest.main()
