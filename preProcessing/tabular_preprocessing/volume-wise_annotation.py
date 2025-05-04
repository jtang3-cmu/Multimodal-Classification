import os
import pandas as pd
from datetime import datetime

# 1. Read the entire Excel workbook (all sheets) and extract the 'annotations' sheet
# file_path = r"D:\cleaning_GUI_annotated_Data\new_data_tab_data.xlsx"
file_path = r"D:\cleaning_GUI_annotated_Data\tab_data_annotated_pats.xlsx"
dfs_dict  = pd.read_excel(file_path, sheet_name=None)
df        = dfs_dict['annotations']

# 2. Specify the root directory for OCT images
# base_dir = r"D:\cleaning_GUI_annotated_Data\New_Data"
base_dir = r"D:\cleaning_GUI_annotated_Data\Cirrus_OCT_Imaging_Data"

# 3. List the columns that serve as stage annotations
stage_cols = ['Early AMD','Int AMD','GA','Wet','Scar','Not AMD']

# 4. Convert these date columns to pandas datetime, ensure there's a 'baseline_stage' column
for c in stage_cols:
    df[c] = pd.to_datetime(df[c], errors='coerce')
# Assume the annotations sheet already has a 'baseline_stage' column if needed

# 5. Define a helper: given an annotation row and a visit date, return the stage at that scan
def get_stage_for_date(row, visit_dt):
    # Collect all events as (date, label)
    raw = [(row[c], c) for c in stage_cols if pd.notnull(row[c])]
    # Group by date: date -> set of labels
    by_date = {}
    for dt, lbl in raw:
        by_date.setdefault(dt, set()).add(lbl)
    # Keep only events on or before the visit date, resolve same-day conflicts
    events = []
    for dt, labels in by_date.items():
        if dt <= visit_dt:
            if labels == {'Wet', 'Scar'}:
                # If only Wet & Scar on the same day, resolve to Scar
                resolved = 'Scar'
            elif len(labels) == 1:
                resolved = next(iter(labels))
            else:
                # For other multi-label conflicts, pick the first alphabetically (or define your logic)
                resolved = sorted(labels)[0]
            events.append((dt, resolved))
    if not events:
        return None
    # The latest event determines the current stage
    latest_dt = max(dt for dt, _ in events)
    for dt, lbl in events:
        if dt == latest_dt:
            return lbl

# 6. Iterate through each patient / laterality / visit folder and assign stage
records = []
for _, row in df.iterrows():
    pid = str(row['research_id']).zfill(9)
    lat = row['laterality']  # 'L' or 'R'
    eye_dir = os.path.join(base_dir, pid, lat)
    if not os.path.isdir(eye_dir):
        continue

    for visit in os.listdir(eye_dir):
        path = os.path.join(eye_dir, visit)
        if not os.path.isdir(path):
            continue
        # Only process folders named as YYYYMMDD
        try:
            vdt = datetime.strptime(visit, '%Y%m%d')
        except ValueError:
            continue

        stage = get_stage_for_date(row, vdt)
        records.append({
            'research_id':    pid,
            'laterality':     lat,
            'baseline_stage': row.get('baseline_stage', None),
            'visit_date':     visit,      # YYYYMMDD
            'visit_date_dt':  vdt,        # datetime
            'stage':          stage
        })

# 7. Output to a new Excel file
out_df = pd.DataFrame(records)
out_df.sort_values(['research_id','laterality','visit_date_dt'], inplace=True)
out_df.to_excel(
    r"D:\cleaning_GUI_annotated_Data\volume_labels.xlsx",
    index=False
)

print("Done! The labeled results have been written to:")
print(r"D:\cleaning_GUI_annotated_Data\volume_labels.xlsx")
