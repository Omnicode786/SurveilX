# Accuracy campaign

Paired development evaluation on frozen splits; candidates stay inactive

| Candidate | State | Metric | Parent | Candidate | Change |
|---|---|---|---:|---:|---:|
| accuracy-g2-fall | completed | accuracy | 0.5000 | 1.0000 | +0.5000 |
| accuracy-g2-entity-v1 | completed | accuracy | 0.6500 | 0.7000 | +0.0500 |
| accuracy-g2-entity-v2 | completed | accuracy | 1.0000 | 1.0000 | +0.0000 |
| accuracy-g2-ppe-scratch | completed | map50 | 0.0136 | 0.0432 | +0.0296 |
| accuracy-g2-ppe-yolo | completed | map50 | 0.0477 | 0.1658 | +0.1181 |
| accuracy-g2-weapons-scratch | completed | map50 | 0.0098 | 0.0230 | +0.0132 |
| accuracy-g2-weapons-yolo | completed | map50 | 0.2130 | 0.3230 | +0.1100 |
| accuracy-g2-fire-scratch | completed | map50 | 0.0884 | 0.2106 | +0.1222 |
| accuracy-g2-fire-yolo | completed | map50 | 0.2112 | 0.3190 | +0.1078 |
| accuracy-g2-person-scratch | completed | map50 | 0.6249 | 0.7177 | +0.0928 |
| accuracy-g2-person-yolo | completed | map50 | 0.8645 | 0.9427 | +0.0782 |
| accuracy-g2-person-baseline | completed | map50 | 0.8493 | 0.9392 | +0.0899 |

The accompanying JSON report records class results and split coverage, including regressions and absent classes.
Test scores are reports only; checkpoint selection uses validation. Repeated development tests do not replace independent acceptance.

## Remaining data and hardware gaps

- General 80-class/domain-inherited YOLO copies lack local labeled 80-class domain evaluation data.
- Fighting, theft, traffic collisions and industrial hazard events lack trained real-data candidates.
- SH17 has sparse PPE labels and classes absent from held-out splits; coverage is reported per class.
- These development sets do not establish independent camera/site acceptance or deployment readiness.
- Entity motion data are synthetic; improved synthetic accuracy does not validate real video events.
- Only CPU execution is available here; CUDA/FPGA performance is unverified.

## Class-level results

### accuracy-g2-fall

| Class | Metric | Parent | Candidate | Change | Train labels | Test labels |
|---|---|---:|---:|---:|---:|---:|
| normal | recall | 0.0000 | 1.0000 | 1.0000 | 5 | 5 |
| fall | recall | 1.0000 | 1.0000 | 0.0000 | 5 | 5 |

### accuracy-g2-entity-v1

| Class | Metric | Parent | Candidate | Change | Train labels | Test labels |
|---|---|---:|---:|---:|---:|---:|
| slow_motion | recall | 0.6500 | 0.6500 | 0.0000 | 20 | 20 |
| fast_motion | recall | 0.6500 | 0.7500 | 0.1000 | 20 | 20 |

### accuracy-g2-entity-v2

| Class | Metric | Parent | Candidate | Change | Train labels | Test labels |
|---|---|---:|---:|---:|---:|---:|
| slow_motion | recall | 1.0000 | 1.0000 | 0.0000 | 50 | 50 |
| fast_motion | recall | 1.0000 | 1.0000 | 0.0000 | 50 | 50 |

### accuracy-g2-ppe-scratch

| Class | Metric | Parent | Candidate | Change | Train labels | Test labels |
|---|---|---:|---:|---:|---:|---:|
| person | ap50 | 0.1622 | 0.2693 | 0.1071 | 233 | 58 |
| ear | ap50 | 0.0000 | 0.0000 | 0.0000 | 131 | 44 |
| earmuffs | ap50 | 0.0000 | 0.0000 | 0.0000 | 2 | 1 |
| face | ap50 | 0.0000 | 0.1718 | 0.1718 | 164 | 36 |
| face_guard | ap50 | not evaluated | not evaluated | not evaluated | 0 | 0 |
| face_mask | ap50 | 0.0000 | 0.0000 | 0.0000 | 6 | 3 |
| foot | ap50 | 0.0000 | 0.0000 | 0.0000 | 11 | 10 |
| tools | ap50 | 0.0000 | 0.0000 | 0.0000 | 57 | 37 |
| glasses | ap50 | 0.0000 | 0.0000 | 0.0000 | 34 | 14 |
| gloves | ap50 | 0.0000 | 0.0000 | 0.0000 | 54 | 10 |
| helmet | ap50 | 0.0000 | 0.0000 | 0.0000 | 13 | 5 |
| hands | ap50 | 0.0100 | 0.0314 | 0.0214 | 265 | 73 |
| head | ap50 | 0.0313 | 0.1752 | 0.1438 | 212 | 53 |
| medical_suit | ap50 | 0.0000 | 0.0000 | 0.0000 | 1 | 4 |
| shoes | ap50 | 0.0000 | 0.0000 | 0.0000 | 93 | 14 |
| safety_suit | ap50 | not evaluated | not evaluated | not evaluated | 5 | 0 |
| safety_vest | ap50 | 0.0000 | 0.0000 | 0.0000 | 10 | 1 |

### accuracy-g2-ppe-yolo

| Class | Metric | Parent | Candidate | Change | Train labels | Test labels |
|---|---|---:|---:|---:|---:|---:|
| person | ap50 | 0.7152 | 0.8515 | 0.1363 | 233 | 58 |
| ear | ap50 | 0.0000 | 0.0000 | 0.0000 | 131 | 44 |
| earmuffs | ap50 | 0.0000 | 0.0000 | 0.0000 | 2 | 1 |
| face | ap50 | 0.0000 | 0.6224 | 0.6224 | 164 | 36 |
| face_guard | ap50 | not evaluated | not evaluated | not evaluated | 0 | 0 |
| face_mask | ap50 | 0.0000 | 0.0000 | 0.0000 | 6 | 3 |
| foot | ap50 | 0.0000 | 0.0000 | 0.0000 | 11 | 10 |
| tools | ap50 | 0.0000 | 0.0000 | 0.0000 | 57 | 37 |
| glasses | ap50 | 0.0000 | 0.0000 | 0.0000 | 34 | 14 |
| gloves | ap50 | 0.0000 | 0.0000 | 0.0000 | 54 | 10 |
| helmet | ap50 | 0.0000 | 0.0000 | 0.0000 | 13 | 5 |
| hands | ap50 | 0.0000 | 0.2797 | 0.2797 | 265 | 73 |
| head | ap50 | 0.0000 | 0.7327 | 0.7327 | 212 | 53 |
| medical_suit | ap50 | 0.0000 | 0.0000 | 0.0000 | 1 | 4 |
| shoes | ap50 | 0.0000 | 0.0000 | 0.0000 | 93 | 14 |
| safety_suit | ap50 | not evaluated | not evaluated | not evaluated | 5 | 0 |
| safety_vest | ap50 | 0.0000 | 0.0000 | 0.0000 | 10 | 1 |

### accuracy-g2-weapons-scratch

| Class | Metric | Parent | Candidate | Change | Train labels | Test labels |
|---|---|---:|---:|---:|---:|---:|
| firearm | ap50 | 0.0007 | 0.0009 | 0.0002 | 114 | 33 |
| knife | ap50 | 0.0092 | 0.0209 | 0.0117 | 120 | 34 |
| machete | ap50 | 0.0255 | 0.0680 | 0.0426 | 112 | 34 |
| baseball_bat | ap50 | 0.0037 | 0.0021 | -0.0017 | 97 | 26 |

### accuracy-g2-weapons-yolo

| Class | Metric | Parent | Candidate | Change | Train labels | Test labels |
|---|---|---:|---:|---:|---:|---:|
| firearm | ap50 | 0.1456 | 0.2685 | 0.1229 | 114 | 33 |
| knife | ap50 | 0.0931 | 0.2306 | 0.1375 | 120 | 34 |
| machete | ap50 | 0.2622 | 0.3076 | 0.0455 | 112 | 34 |
| baseball_bat | ap50 | 0.3511 | 0.4853 | 0.1342 | 97 | 26 |

### accuracy-g2-fire-scratch

| Class | Metric | Parent | Candidate | Change | Train labels | Test labels |
|---|---|---:|---:|---:|---:|---:|
| smoke | ap50 | 0.1348 | 0.2120 | 0.0772 | 225 | 52 |
| fire | ap50 | 0.0420 | 0.2092 | 0.1672 | 403 | 120 |

### accuracy-g2-fire-yolo

| Class | Metric | Parent | Candidate | Change | Train labels | Test labels |
|---|---|---:|---:|---:|---:|---:|
| smoke | ap50 | 0.2407 | 0.3508 | 0.1101 | 225 | 52 |
| fire | ap50 | 0.1817 | 0.2873 | 0.1056 | 403 | 120 |

### accuracy-g2-person-scratch

| Class | Metric | Parent | Candidate | Change | Train labels | Test labels |
|---|---|---:|---:|---:|---:|---:|
| person | ap50 | 0.6249 | 0.7177 | 0.0928 | 245 | 58 |

### accuracy-g2-person-yolo

| Class | Metric | Parent | Candidate | Change | Train labels | Test labels |
|---|---|---:|---:|---:|---:|---:|
| person | ap50 | 0.8645 | 0.9427 | 0.0782 | 245 | 58 |

### accuracy-g2-person-baseline

| Class | Metric | Parent | Candidate | Change | Train labels | Test labels |
|---|---|---:|---:|---:|---:|---:|
| person | ap50 | 0.8493 | 0.9392 | 0.0899 | 245 | 58 |

