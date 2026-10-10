# Offline tracking evaluation

Run `.venv/Scripts/python.exe -m scripts.evaluate_tracking input.json new-report.json --iou 0.5`. This reads labeled detections only: no inference, training, downloading or activation. Existing reports cannot be overwritten. The output records the input SHA256, evidence flags, per-sequence and per-class counts and metrics.

Input schema example:

```json
{"schema_version":1,"synthetic":true,"ground_truth_reviewed":false,"sequences":[{"name":"example","frames":[{"index":0,"ground_truth":[{"track_id":"person-a","label":"person","box":[0.1,0.1,0.5,0.5]}],"predictions":[{"track_id":7,"label":"person","box":[0.1,0.1,0.5,0.5]}]}]}]}
```

Use consecutive frame indices and explicitly include empty frames. Boxes are finite normalized xyxy with positive area. IDs must be unique within each frame, retain their class and are scoped to their sequence. Classes are matched exactly. Annotate actual visible objects and review identities independently of predictions before interpreting model performance.

The evaluator uses class-aware IoU matching with previous-frame continuity preference. ID switches compare against the last matched identity across gaps; empty frames reset immediate continuity preference. Global one-to-one identity association counts all eligible IoU pairs. It reports false positives, misses, switches, precision/recall, MOTA, mean matched IoU (MOTP) and IDF1. Undefined ratios are null; MOTA can be negative.

Definitions were checked against primary [TrackEval CLEAR](https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/metrics/clear.py) and [identity metric](https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/metrics/identity.py) implementations. This lightweight development evaluator is **not an official MOTChallenge adapter or a parity certification**: ignored regions, occlusion/visibility protocols, benchmark-specific preprocessing and HOTA are absent. Empty-frame continuity conventions can differ. Use the official adapter for publishable benchmark comparisons.

`reports/tracking-synthetic-evaluation.json` checks a perfect box whose ID changes: two matches, one switch, MOTA/IDF1 0.5. Synthetic tests cover duplicates, wrong classes, gaps, crossing assignment, rectangular global matching and invalid geometry. These prove software correctness on fixtures, not field tracking accuracy. Real identity-labeled footage remains missing.
