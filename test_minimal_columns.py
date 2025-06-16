from pathlib import Path

import openpyxl
import pandas as pd

# Test file path
file_path = Path("test1.xlsx")

# Direct openpyxl approach - use worksheet dimensions
wb = openpyxl.load_workbook(file_path, data_only=True, read_only=True)
ws = wb.active

# Get the actual data range (avoids reading corrupted metadata)
max_row = ws.max_row
max_col = ws.max_column
print(f"Worksheet dimensions: {max_row} rows, {max_col} columns")

data = []
headers = []
for i, row in enumerate(ws.iter_rows(max_row=max_row, values_only=True)):
    if i == 0:
        headers = [
            str(cell) if cell is not None else f"col_{j}" for j, cell in enumerate(row)
        ]
    else:
        data.append(row)

wb.close()
test_df = pd.DataFrame(data, columns=headers)
print(f"Shape: {test_df.shape}")
print(test_df.tail(1))
