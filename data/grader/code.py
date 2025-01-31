from statistics import mean
from typing import Dict, List, Tuple


class GradeProcessor:
    def __init__(self, class_data: Dict[str, Dict[str, List[Tuple[str, float]]]]):
        """
        class_data format:
        {
            'class_name': {
                'section': [(student_id, grade), ...]
            }
        }
        """
        self.class_data = class_data

    def get_top_performers(self, threshold: float = 90.0) -> List[str]:
        """
        Returns student IDs of those who scored above the threshold
        in all their classes, if a stuent has incomplete grades, they are not considered.
        """
        # First, gather all grades for each student
        student_grades = {}
        for class_info in self.class_data.values():
            for section in class_info.values():
                for student_id, grade in section:
                    if student_id not in student_grades:
                        student_grades[student_id] = []
                    student_grades[student_id].append(grade)

        # Then check if all grades are above threshold
        return [
            student_id 
            for student_id, grades in student_grades.items()
            if all(grade >= threshold for grade in grades)
        ]

    def get_student_averages(self) -> Dict[str, float]:
        """
        Calculate average grade for each student across all classes
        """
        # Another complex (but correct) comprehension for comparison
        all_grades = {
            student_id: mean([grade 
                for class_info in self.class_data.values()
                for section in class_info.values()
                for (sid, grade) in section
                if sid == student_id
            ])
            for class_info in self.class_data.values()
            for section in class_info.values()
            for (student_id, _) in section
        }
        return all_grades