# Product development evidence

Artifact integrity and development coverage audit; no training, activation or production acceptance.

A successful training run or aggregate score is not proof of reliable multi-domain detection.

| Family | Imported datasets | Evaluated candidates | Every model class >=0.50 | Implemented policy rules |
|---|---:|---:|---:|---|
| General objects | 0 | 0 | 0 | N/A |
| People | 1 | 10 | 9 | N/A |
| Fire and smoke | 2 | 8 | 0 | N/A |
| Firearms | 1 | 9 | 0 | N/A |
| Fighting | 1 | 2 | 2 | N/A |
| Falls | 1 | 2 | 1 | N/A |
| Possible theft events | 0 | 0 | 0 | N/A |
| Traffic collisions | 0 | 0 | 0 | N/A |
| Traffic rule review | N/A | N/A | N/A | wrong_way, calibrated_speed, signal_stop_line |
| Protective equipment | 2 | 9 | 0 | N/A |
| PPE compliance review | N/A | N/A | N/A | possible_missing_helmet, possible_missing_vest |
| Industrial hazard review | N/A | N/A | N/A | restricted_zone, configured_proximity, machine_state, blocked_exit |

## Interrupted-run disposition

- `accuracy-g3-weapons-scratch`: preserved; replacement `accuracy-g3b-weapons-scratch` complete and intact: True.
- `accuracy-ppe-g3-scratch`: preserved; replacement `accuracy-ppe-g3b-scratch` complete and intact: True.

Detailed scores, source limits, hashes and errors are in product-readiness.json. Inherited models without matching local data are not local training evidence.
