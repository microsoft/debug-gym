import unittest

from scientific_calculator_code import PlusTool


class TestPlusTool(unittest.TestCase):

    def setUp(self):
        self.plus_tool = PlusTool()

    def test_use(self):
        self.assertEqual(self.plus_tool.use("```plus 1 2```"), 3)
        self.assertEqual(self.plus_tool.use("```plus 3 4```"), 7)
        self.assertEqual(self.plus_tool.use("```plus -3 3```"), 0)
        self.assertEqual(self.plus_tool.use("```plus 1 2 3```"), "SyntaxError: invalid syntax.")
        self.assertEqual(self.plus_tool.use("```plus 1 a```"), "ValueError: invalid value.")
