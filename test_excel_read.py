from pathlib import Path

import openpyxl
import pandas as pd

# Test file path
file_path = Path('/home/jovyan/DU/LayoutLM_annotation/evaluation_test/evaluation_testset_Jun_16/1-10DT9W8W_1252183212_8_17.xlsx')

# Direct openpyxl approach
wb = openpyxl.load_workbook(file_path, data_only=True, read_only=True)
ws = wb.active
data = []
headers = []
for i, row in enumerate(ws.iter_rows(values_only=True)):
    if i == 0:
        headers = [str(cell) if cell is not None else f"col_{j}" for j, cell in enumerate(row)]
    else:
        data.append(row)
    if i > 10:  # Just test first 10 rows
        break
wb.close()
test_df = pd.DataFrame(data, columns=headers)
print(f"Shape: {test_df.shape}")
print(test_df.head())