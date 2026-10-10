# G4 dataset and historical assignment comparison

Descriptive frozen-data and SVA-v1 assignment audit; correlation does not establish causation or independent field accuracy.

All image SHA256 values checked against manifests. No training or downloads.

Sizes use actual g4 square stretching. Sub-stride size is a risk indicator, not a detection impossibility. Letterboxing preserves shape but can reduce small-object pixels at equal canvas size.

## dfire-development-v1 (384 px)

Train: 384 images, 64 empty, 628 objects; 2 classes, 3 recorded groups. Object imbalance max/min: 1.79.

| Class | Train objects / images | Val / cal / test objects | Median short side px | <8 px | Aspect >4 | Lost objects | Positive assignments |
|---|---:|---:|---:|---:|---:|---:|---:|
| smoke | 225 / 214 | 61 / 57 / 52 | 107.52 | 2 | 4 | 4 | 1958 |
| fire | 403 / 213 | 72 / 64 / 120 | 19.20 | 41 | 7 | 0 | 3596 |

Assignment totals: {"positive_count": 5554, "positive_counts_per_object": {"min": 0.0, "p10": 9.0, "p25": 9.0, "median": 9.0, "p75": 9.0, "p90": 9.0, "max": 12.0}, "positive_counts_per_level": [3999, 805, 750], "objects_lost_to_conflicts": 4, "objects_with_fewer_than_three_positives": 5, "positive_points_outside_assigned_box": 1273}

## dfire-development-v2 (384 px)

Train: 1284 images, 239 empty, 2065 objects; 2 classes, 3 recorded groups. Object imbalance max/min: 1.61.

| Class | Train objects / images | Val / cal / test objects | Median short side px | <8 px | Aspect >4 | Lost objects | Positive assignments |
|---|---:|---:|---:|---:|---:|---:|---:|
| smoke | 790 / 737 | 61 / 57 / 52 | 113.77 | 4 | 16 | 11 | 6752 |
| fire | 1275 / 639 | 72 / 64 / 120 | 21.45 | 122 | 23 | 0 | 11382 |

Assignment totals: {"positive_count": 18134, "positive_counts_per_object": {"min": 0.0, "p10": 9.0, "p25": 9.0, "median": 9.0, "p75": 9.0, "p90": 9.0, "max": 12.0}, "positive_counts_per_level": [12537, 3046, 2551], "objects_lost_to_conflicts": 11, "objects_with_fewer_than_three_positives": 14, "positive_points_outside_assigned_box": 3421}

## dangerous-items-development-v1 (384 px)

Train: 399 images, 0 empty, 443 objects; 4 classes, 399 recorded groups. Object imbalance max/min: 1.24.

| Class | Train objects / images | Val / cal / test objects | Median short side px | <8 px | Aspect >4 | Lost objects | Positive assignments |
|---|---:|---:|---:|---:|---:|---:|---:|
| firearm | 114 / 106 | 26 / 18 / 33 | 83.52 | 0 | 12 | 0 | 1038 |
| knife | 120 / 111 | 28 / 18 / 34 | 93.61 | 0 | 17 | 0 | 1081 |
| machete | 112 / 104 | 27 / 17 / 34 | 96.96 | 0 | 19 | 0 | 1008 |
| baseball_bat | 97 / 96 | 23 / 16 / 26 | 99.20 | 0 | 15 | 0 | 879 |

Assignment totals: {"positive_count": 4006, "positive_counts_per_object": {"min": 3.0, "p10": 9.0, "p25": 9.0, "median": 9.0, "p75": 9.0, "p90": 9.0, "max": 12.0}, "positive_counts_per_level": [1041, 1560, 1405], "objects_lost_to_conflicts": 0, "objects_with_fewer_than_three_positives": 0, "positive_points_outside_assigned_box": 231}

## sh17-development-v2 (512 px)

Train: 631 images, 0 empty, 8062 objects; 17 classes, 205 recorded groups. Object imbalance max/min: 38.38.

| Class | Train objects / images | Val / cal / test objects | Median short side px | <8 px | Aspect >4 | Lost objects | Positive assignments |
|---|---:|---:|---:|---:|---:|---:|---:|
| person | 1305 / 608 | 316 / 301 / 245 | 92.73 | 53 | 159 | 187 | 8448 |
| ear | 577 / 324 | 129 / 124 / 124 | 7.11 | 323 | 24 | 4 | 5087 |
| earmuffs | 135 / 100 | 30 / 34 / 13 | 42.63 | 5 | 1 | 8 | 983 |
| face | 649 / 388 | 176 / 124 / 101 | 31.54 | 86 | 4 | 62 | 4384 |
| face_guard | 74 / 72 | 15 / 20 / 6 | 62.18 | 3 | 0 | 3 | 543 |
| face_mask | 202 / 100 | 34 / 40 / 42 | 20.98 | 44 | 0 | 13 | 1321 |
| foot | 180 / 100 | 33 / 49 / 60 | 18.55 | 36 | 3 | 6 | 1434 |
| tools | 574 / 197 | 59 / 61 / 84 | 12.21 | 200 | 212 | 97 | 3343 |
| glasses | 179 / 154 | 47 / 26 / 62 | 22.28 | 24 | 9 | 3 | 1366 |
| gloves | 461 / 205 | 114 / 105 / 92 | 17.76 | 134 | 0 | 7 | 3750 |
| helmet | 261 / 103 | 96 / 70 / 42 | 12.26 | 84 | 0 | 43 | 1431 |
| hands | 1104 / 414 | 279 / 230 / 205 | 21.62 | 198 | 5 | 14 | 9251 |
| head | 1184 / 566 | 299 / 266 / 226 | 33.68 | 188 | 0 | 311 | 5082 |
| medical_suit | 34 / 27 | 34 / 2 / 44 | 183.96 | 0 | 0 | 0 | 287 |
| shoes | 742 / 231 | 148 / 162 / 105 | 16.08 | 188 | 8 | 63 | 5216 |
| safety_suit | 168 / 100 | 28 / 25 / 6 | 66.48 | 2 | 13 | 7 | 1188 |
| safety_vest | 233 / 87 | 56 / 51 / 45 | 22.48 | 38 | 3 | 48 | 1291 |

Assignment totals: {"positive_count": 54405, "positive_counts_per_object": {"min": 0.0, "p10": 0.0, "p25": 5.0, "median": 9.0, "p75": 9.0, "p90": 9.0, "max": 16.0}, "positive_counts_per_level": [44575, 5873, 3957], "objects_lost_to_conflicts": 876, "objects_with_fewer_than_three_positives": 1180, "positive_points_outside_assigned_box": 14821}

## Fire expansion control

{
  "all_old_records_retained_identically": true,
  "old_record_count": 672,
  "additions_by_split": {
    "train": 900
  },
  "interpretation": "Only fire received additional images in g4b. Data, focal loss, batch size and training device changed together; focal effect is not isolated."
}

## Storage cleanup candidates

Check that no training/resume process uses these directories before removal. Original dataset versions remain referenced by historical artifacts; retain them.

Rebuildable bytes: 1,128,703,732. This script deletes nothing.

- `C:\Users\muzam\OneDrive\Desktop\Programming\SurveilX\data\runs\accuracy-g2-fire-yolo\dataset`: 15,427,388 bytes; weights hash valid: True.
- `C:\Users\muzam\OneDrive\Desktop\Programming\SurveilX\data\runs\accuracy-g2-person-baseline\dataset`: 22,258,585 bytes; weights hash valid: True.
- `C:\Users\muzam\OneDrive\Desktop\Programming\SurveilX\data\runs\accuracy-g2-person-yolo\dataset`: 22,258,101 bytes; weights hash valid: True.
- `C:\Users\muzam\OneDrive\Desktop\Programming\SurveilX\data\runs\accuracy-g2-ppe-yolo\dataset`: 9,062,334 bytes; weights hash valid: True.
- `C:\Users\muzam\OneDrive\Desktop\Programming\SurveilX\data\runs\accuracy-g2-weapons-yolo\dataset`: 14,510,778 bytes; weights hash valid: True.
- `C:\Users\muzam\OneDrive\Desktop\Programming\SurveilX\data\runs\accuracy-g3-fire-yolo\dataset`: 29,081,215 bytes; weights hash valid: True.
- `C:\Users\muzam\OneDrive\Desktop\Programming\SurveilX\data\runs\accuracy-g3-weapons-yolo\dataset`: 60,173,532 bytes; weights hash valid: True.
- `C:\Users\muzam\OneDrive\Desktop\Programming\SurveilX\data\runs\accuracy-ppe-g3-yolo\dataset`: 109,769,644 bytes; weights hash valid: True.
- `C:\Users\muzam\OneDrive\Desktop\Programming\SurveilX\data\runs\cuda-g4-fire-yolo\dataset`: 65,823,915 bytes; weights hash valid: True.
- `C:\Users\muzam\OneDrive\Desktop\Programming\SurveilX\data\runs\cuda-g4-ppe-yolo\dataset`: 109,766,528 bytes; weights hash valid: True.
- `C:\Users\muzam\OneDrive\Desktop\Programming\SurveilX\data\runs\cuda-g4-weapons-yolo\dataset`: 31,088,743 bytes; weights hash valid: True.
- `C:\Users\muzam\OneDrive\Desktop\Programming\SurveilX\data\runs\dangerous-items-g1-yolo-rai\dataset`: 60,175,020 bytes; weights hash valid: True.
- `C:\Users\muzam\OneDrive\Desktop\Programming\SurveilX\data\runs\dfire-g1-yolo-rai\dataset`: 73,705,743 bytes; weights hash valid: True.
- `C:\Users\muzam\OneDrive\Desktop\Programming\SurveilX\data\runs\sh17-g2-yolo-rai\dataset`: 402,897,475 bytes; weights hash valid: True.
- `C:\Users\muzam\OneDrive\Desktop\Programming\SurveilX\data\runs\yolo-baseline-pennfudan-v1\dataset`: 51,352,668 bytes; weights hash valid: True.
- `C:\Users\muzam\OneDrive\Desktop\Programming\SurveilX\data\runs\yolo-rai-pennfudan-v1\dataset`: 51,352,063 bytes; weights hash valid: True.


Cleanup completed after this discovery report: 15 verified derived directories removed, 1,054,997,989 bytes recovered. The uncertain older D-Fire cache was retained. See g4-cleanup-journal.json and g4-final-integrity.json for final state.
