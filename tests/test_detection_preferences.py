from types import SimpleNamespace
import pytest
from sqlalchemy import select
from test_api import login
from test_runtime import make_capture
from surveilx.detection_preferences import DetectionSelection, category
from surveilx.vision import Detection


def test_label_aliases_and_groups_do_not_conflate_objects():
    options=DetectionSelection(disabled_labels=['People','Person','Fire'])
    assert options.disabled_labels==['fire','person']
    assert not options.enabled('pedestrian')
    assert options.enabled('firearm')
    assert category('bear')=='Animals'
    assert category('carrot')=='Other objects'
    assert category('handbag')=='Other objects'
    with pytest.raises(ValueError):
        DetectionSelection(disabled_labels=['background'])


def test_disabled_events_keep_original_scores_without_renormalization():
    options=DetectionSelection(disabled_labels=['fighting'])
    raw={'decision':'experimental_evidence','calibrated':True,'classes':['normal','fighting','fall'], 'probabilities':[.1,.8,.1]}
    filtered=options.event(raw)
    assert filtered['classes']==['normal','fall']
    assert filtered['probabilities']==[.1,.1]
    assert filtered['excluded_probability_mass']==.8
    assert raw['probabilities']==[.1,.8,.1]
    assert DetectionSelection(disabled_labels=['fighting','fall']).event(raw) is None
    from surveilx.policies import PolicyEngine
    policy=PolicyEngine({'rules':[{'id':'fight','kind':'scene_event','labels':['fighting'],'confidence':.6,'duration':.5,'min_observations':2}]})
    for now in range(5):
        assert policy.evaluate_event(filtered,now,'fixture')==[]
    assert not policy.states


def test_selection_api_persists_and_protects_operator_permissions(client):
    login(client)
    camera=client.post('/api/cameras',json={'name':'Selection test','source':'demo://0'}).json()
    endpoint=f'/api/cameras/{camera["id"]}/detection-options'
    assert client.get(endpoint).json()['disabled_labels']==[]
    response=client.put(endpoint,json={'disabled_labels':['Persons','fire']})
    assert response.status_code==200
    assert client.get(endpoint).json()['disabled_labels']==['fire','person']
    assert any(row['id']=='person' and not row['enabled'] for row in response.json()['options'])
    from surveilx import api
    api.runtime.restart_camera(camera['id'])
    assert not api.runtime.cameras[camera['id']].detection_selection.enabled('person')
    assert client.put(endpoint,json={'disabled_labels':['background']}).status_code==422
    client.post('/api/users',json={'username':'reader','password':'reader-password-long','role':'viewer'})
    client.post('/api/auth/logout')
    client.post('/api/auth/login',json={'username':'reader','password':'reader-password-long'})
    assert client.get(endpoint).status_code==200
    assert client.put(endpoint,json={'disabled_labels':[]}).status_code==403


def test_disabled_person_never_enters_tracks_policies_or_new_evidence(client):
    runtime,capture=make_capture(client)
    calls=[]
    def infer(*args):
        calls.append(1)
        return [Detection([.1,.1,.4,.9],'person',.9),Detection([.5,.1,.8,.7],'fire',.9)]
    runtime.synthetic_detector=SimpleNamespace(name='fixture',version='selection-fixture',manifest={'classes':['person','fire']},infer=infer)
    runtime.set_detection_selection(capture.id,DetectionSelection(disabled_labels=['person']))
    runtime.infer(capture)
    assert [row['label'] for row in capture.outputs]==['fire']
    assert not capture.entity_history  # No temporal consumer: no history encoding.
    from surveilx.database import Incident,Record,transaction
    with transaction() as session:
        incidents=session.scalars(select(Incident)).all()
        assert incidents and all(row.details['observed_labels']==['fire'] for row in incidents)
        records=session.scalars(select(Record).where(Record.kind=='decision')).all()
        assert all([item['label']for item in row.payload['result']]==['fire']for row in records)
    runtime.set_detection_selection(capture.id,DetectionSelection(disabled_labels=['person','fire']))
    capture.sequence+=1
    runtime.infer(capture)
    assert len(calls)==1  # All known classes off: don't execute this model.
    assert not capture.outputs and not capture.candidate_since
    assert capture.secondary_result is None
    with transaction() as session:
        assert len(session.scalars(select(Incident)).all())==len(incidents)  # Old evidence retained.


def test_disabled_labels_apply_to_real_baseline_fallback(client):
    runtime,capture=make_capture(client)
    capture.synthetic=False
    def fail(*args):raise RuntimeError('fixture failure')
    runtime.yolo=SimpleNamespace(name='bad-fixture',infer=fail)
    runtime.baseline=SimpleNamespace(name='fallback-fixture',infer=lambda *args:[Detection([.1,.1,.9,.9],'person',.9)])
    runtime.set_detection_selection(capture.id,DetectionSelection(disabled_labels=['person']))
    runtime.infer(capture)
    assert not capture.outputs
    assert 'baseline fallback' in runtime.error


def test_disabling_helmet_does_not_turn_into_missing_helmet_alert():
    from surveilx.policies import PolicyEngine
    rule=PolicyEngine({'rules':[{'id':'ppe','kind':'possible_missing_helmet','duration':.5,'min_observations':2}]})
    outputs=[{'label':'person','box':[.1,.1,.9,.9],'score':.95,'track_id':1},
             {'label':'head','box':[.3,.2,.5,.4],'score':.95,'track_id':2}]
    for now in range(5):
        assert rule.evaluate(outputs,now,'fixture',disabled_labels=['helmet'])==[]
    assert not rule.states
