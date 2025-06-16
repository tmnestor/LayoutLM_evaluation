from pathlib import Path

import openpyxl
import pandas as pd

# Test file path
file_path = Path("test1.xlsx")

# Direct openpyxl approach - read until corruption detected
wb = openpyxl.load_workbook(file_path, data_only=True, read_only=True)
ws = wb.active

data = []
headers = []
row_num = 0

try:
    for row in ws.iter_rows(values_only=True):
        if row_num == 0:
            headers = [
                str(cell) if cell is not None else f"col_{j}"
                for j, cell in enumerate(row)
            ]
        else:
            data.append(row)
        row_num += 1

        # Safety limit to prevent infinite loops
        if row_num > 1000:
            break

except Exception as e:
    print(f"Stopped reading at row {row_num} due to corruption: {str(e)}")

wb.close()
test_df = pd.DataFrame(data, columns=headers)
print(f"Successfully read {row_num} rows")
print(f"Shape: {test_df.shape}")
print(test_df.tail(1))
