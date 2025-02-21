import unittest

from grader_code import GradeProcessor


class TestGradeProcessor(unittest.TestCase):
    def setUp(self):
        # Test data with multiple classes, sections, and students
        self.test_data = {
            'Math': {
                'Section1': [
                    ('student1', 95.0),
                    ('student2', 88.0),
                    ('student3', 92.0)
                ],
                'Section2': [
                    ('student4', 91.0),
                    ('student5', 89.0)
                ]
            },
            'Physics': {
                'Section1': [
                    ('student1', 89.0),  # Note: same student1 in different class
                    ('student3', 94.0),
                    ('student5', 93.0)
                ]
            }
        }
        self.processor = GradeProcessor(self.test_data)

    def test_top_performers(self):
        """
        When a student's score is incomplete, they should not be considered
        """
        # With threshold 90, we expect student3 only
        # (only student with all grades >= 90)
        top_students = self.processor.get_top_performers(threshold=90.0)
        
        # This is what we actually want
        self.assertEqual(
            set(top_students),
            {'student3'},  # Only student3 has all grades >= 90
            "Should only return students with ALL grades above threshold"
        )

    def test_student_averages(self):
        """
        Test the working get_student_averages() method for comparison
        """
        averages = self.processor.get_student_averages()
        
        # Test specific average calculations
        self.assertAlmostEqual(averages['student1'], 92.0)  # (95 + 89) / 2
        self.assertEqual(averages['student2'], 88.0)  # Single grade
        self.assertAlmostEqual(averages['student3'], 93.0)  # (92 + 94) / 2

    def test_edge_cases(self):
        # Empty data
        empty_processor = GradeProcessor({})
        self.assertEqual(empty_processor.get_top_performers(), [])
        
        # Single grade at exactly threshold
        threshold_data = {
            'Math': {
                'Section1': [('student1', 90.0)]
            }
        }
        threshold_processor = GradeProcessor(threshold_data)
        self.assertEqual(
            threshold_processor.get_top_performers(threshold=90.0),
            ['student1']
        )

if __name__ == '__main__':
    unittest.main()