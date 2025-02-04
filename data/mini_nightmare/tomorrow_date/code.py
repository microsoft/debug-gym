from datetime import datetime


class TomorrowDate:
    def __init__(self):
        self.supported_formats = ['%Y-%m-%d', '%m-%d-%Y', '%m.%d.%Y', '%m %d %Y', '%m %d %y']

    def find_dates(self, input_string):
        for format in self.supported_formats:
            try:
                date = datetime.strptime(input_string, format).date()
                return date
            except ValueError:
                pass
        return None
    
    def date_plus_one(self, today):
        year, month, day = today.year, today.month, today.day
        if month == 2:
            if day == 28:
                if year % 4 == 0:
                    return datetime(year, month, day + 1).date()
                else:
                    return datetime(year, month + 1, 1).date()
            else:
                return datetime(year, month, day + 1).date()
            
        if month in [1, 3, 5, 7, 8, 10, 12]:
            if day == 31:
                if month == 12:
                    return datetime(year + 1, 1, 1).date()
                else:
                    return datetime(year, month + 1, 1).date()
            else:
                return datetime(year, month, day + 1).date()
        else:
            if day == 30:
                return datetime(year, month + 1, 1).date()
            else:
                return datetime(year, month, day + 1).date()
            
    def tomorrow(self, input_string):
        input_string = input_string.strip()
        try:
            today = self.find_dates(input_string)
            if today is None:
                return None
        except ValueError:
            return None
        try:
            output = self.date_plus_one(today)
        except ValueError:
            return None
        return output
