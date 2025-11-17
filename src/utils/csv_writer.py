import csv
import os
from datetime import datetime

class CSVLogger:
    def __init__(self, filename=None):
        if filename is None:
            filename = f"angles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.filename = filename
        self.file_exists = os.path.exists(filename)
        self.file = open(filename, 'a', newline='')
        self.writer = None

    def write_header(self, keys):
        if not self.file_exists:
            self.writer = csv.DictWriter(self.file, fieldnames=['timestamp'] + keys)
            self.writer.writeheader()
        else:
            self.writer = csv.DictWriter(self.file, fieldnames=['timestamp'] + keys)

    def log(self, angles_dict):
        if self.writer is None:
            self.write_header(list(angles_dict.keys()))
        row = {'timestamp': datetime.now().isoformat()}
        row.update(angles_dict)
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()
