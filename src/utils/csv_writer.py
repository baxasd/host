"""Simple CSV logger for angle/time-series data.

This utility opens a CSV file in append mode and writes a header the
first time it runs. Each row contains an ISO timestamp followed by the
angle values. The class keeps the file open to reduce IO overhead and
flushes on each write to minimize data loss in case of a crash.
"""

import csv
import os
from datetime import datetime


class CSVLogger:
    def __init__(self, filename=None):
        # Default filename includes a timestamp for uniqueness
        if filename is None:
            filename = f"angles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.filename = filename
        self.file_exists = os.path.exists(filename)
        # Open the file in append mode so multiple runs do not overwrite data
        self.file = open(filename, 'a', newline='')
        self.writer = None

    def write_header(self, keys):
        # Create a DictWriter and write header only the first time
        if not self.file_exists:
            self.writer = csv.DictWriter(self.file, fieldnames=['timestamp'] + keys)
            self.writer.writeheader()
        else:
            self.writer = csv.DictWriter(self.file, fieldnames=['timestamp'] + keys)

    def log(self, angles_dict):
        # Lazily write header on first log call when angle keys are known
        if self.writer is None:
            self.write_header(list(angles_dict.keys()))
        row = {'timestamp': datetime.now().isoformat()}
        row.update(angles_dict)
        self.writer.writerow(row)
        # Flush to keep the file consistent if the process stops unexpectedly
        self.file.flush()

    def close(self):
        # Close underlying file handle
        self.file.close()
