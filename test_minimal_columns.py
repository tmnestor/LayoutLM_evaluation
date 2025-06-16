from pathlib import Path

import openpyxl
import pandas as pd

# Test file path
file_path = Path("test.xlsx")

# Direct openpyxl approach with dynamic row detection
wb = openpyxl.load_workbook(file_path, data_only=True, read_only=True)
ws = wb.active

# First pass: count actual data rows
actual_rows = 0
for row in ws.iter_rows(values_only=True):
    if any(cell is not None for cell in row):  # Check if row has any data
        actual_rows += 1
    else:
        break  # Stop at first empty row

wb.close()

# Second pass: read data with correct row count
wb = openpyxl.load_workbook(file_path, data_only=True, read_only=True)
ws = wb.active
data = []
headers = []
for i, row in enumerate(ws.iter_rows(values_only=True)):
    if i == 0:
        headers = [
            str(cell) if cell is not None else f"col_{j}" for j, cell in enumerate(row)
        ]
    else:
        data.append(row)
    if i >= actual_rows - 1:  # Stop at actual data end
        break

wb.close()
test_df = pd.DataFrame(data, columns=headers)
print(f"Detected {actual_rows} rows")
print(f"Shape: {test_df.shape}")
print(test_df.tail(1))
