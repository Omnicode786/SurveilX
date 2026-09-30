# Accuracy data audit

Label counts describe coverage, not statistical independence or sufficient sample size

Object sizes use training labels only. Sub-stride objects flag a resolution constraint; this is not a measured detection limit.

## urfall-development-v1

| Class | Train | Validation | Calibration | Test | Median short side at 256 px |
|---|---:|---:|---:|---:|---:|
| normal | 5 | 5 | 5 | 5 | — |
| fall | 5 | 5 | 5 | 5 | — |

## generated-v1

| Class | Train | Validation | Calibration | Test | Median short side at 256 px |
|---|---:|---:|---:|---:|---:|
| slow_motion | 20 | 20 | 20 | 20 | — |
| fast_motion | 20 | 20 | 20 | 20 | — |

## generated-v2

| Class | Train | Validation | Calibration | Test | Median short side at 256 px |
|---|---:|---:|---:|---:|---:|
| slow_motion | 50 | 50 | 50 | 50 | — |
| fast_motion | 50 | 50 | 50 | 50 | — |

## sh17-development-v1

| Class | Train | Validation | Calibration | Test | Median short side at 256 px |
|---|---:|---:|---:|---:|---:|
| person | 233 | 60 | 50 | 58 | 90.0 |
| ear | 131 | 22 | 31 | 44 | 5.2 |
| earmuffs | 2 | 0 | 0 | 1 | 10.2 |
| face | 164 | 40 | 24 | 36 | 21.9 |
| face_guard | 0 | 0 | 0 | 0 | — |
| face_mask | 6 | 3 | 2 | 3 | 6.4 |
| foot | 11 | 0 | 4 | 10 | 11.6 |
| tools | 57 | 8 | 12 | 37 | 27.3 |
| glasses | 34 | 10 | 5 | 14 | 13.6 |
| gloves | 54 | 13 | 12 | 10 | 12.9 |
| helmet | 13 | 17 | 8 | 5 | 5.5 |
| hands | 265 | 84 | 44 | 73 | 17.2 |
| head | 212 | 49 | 45 | 53 | 28.1 |
| medical_suit | 1 | 2 | 0 | 4 | 140.8 |
| shoes | 93 | 31 | 9 | 14 | 10.3 |
| safety_suit | 5 | 0 | 0 | 0 | 28.0 |
| safety_vest | 10 | 12 | 3 | 1 | 7.8 |

## dangerous-items-development-v1

| Class | Train | Validation | Calibration | Test | Median short side at 256 px |
|---|---:|---:|---:|---:|---:|
| firearm | 114 | 26 | 18 | 33 | 55.7 |
| knife | 120 | 28 | 18 | 34 | 62.4 |
| machete | 112 | 27 | 17 | 34 | 64.6 |
| baseball_bat | 97 | 23 | 16 | 26 | 66.1 |

## dfire-development-v1

| Class | Train | Validation | Calibration | Test | Median short side at 256 px |
|---|---:|---:|---:|---:|---:|
| smoke | 225 | 61 | 57 | 52 | 71.7 |
| fire | 403 | 72 | 64 | 120 | 12.8 |

## pennfudan

| Class | Train | Validation | Calibration | Test | Median short side at 256 px |
|---|---:|---:|---:|---:|---:|
| person | 245 | 59 | 45 | 58 | 45.7 |
