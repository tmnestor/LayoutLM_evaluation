from pathlib import Path

import openpyxl
import pandas as pd

# Test file path
file_path = Path('test1.xlsx')

# Direct openpyxl approach - stop at first empty row
wb = openpyxl.load_workbook(file_path, data_only=True, read_only=True)
ws = wb.active
data = []
headers = []
for i, row in enumerate(ws.iter_rows(values_only=True)):
    if i == 0:
        headers = [str(cell) if cell is not None else f"col_{j}" for j, cell in enumerate(row)]
    else:
        # Stop if we hit a completely empty row (end of actual data)
        if all(cell is None or cell == '' for cell in row):
            break
        data.append(row)
wb.close()
test_df = pd.DataFrame(data, columns=headers)
print(f"Shape: {test_df.shape}")
print(test_df.tail(1))