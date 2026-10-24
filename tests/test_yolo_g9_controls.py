from test_api import login
from training.yolo_policy import training_policy


def test_v3_preserves_native_validation_and_pretrained_stem():
    p=training_policy('finetune-v3',True,.0003,48,4)
    assert p['arguments']['freeze']==4
    assert p['arguments']['mosaic']==.15 and p['arguments']['close_mosaic']==5
    assert p['arguments']['warmup_bias_lr']==0 and p['lossless_prepared_images']


def test_admin_runtime_policy_persistence_and_limits(client):
    login(client)
    from surveilx import api
    from surveilx.config import settings
    original=(settings.power_profile,settings.runtime_budget_scale)
    try:
        assert client.patch('/api/settings/runtime',json={'power_profile':'balanced','runtime_budget_scale':1.25}).status_code==200
        assert client.get('/api/settings').json()['power_profile']=='balanced'
        api.restore_runtime_policy()
        assert settings.runtime_budget_scale==1.25
        assert client.patch('/api/settings/runtime',json={'power_profile':'performance','runtime_budget_scale':100}).status_code==422
        client.post('/api/users',json={'username':'reader','password':'reader-password-long','role':'viewer'})
        client.post('/api/auth/logout')
        client.post('/api/auth/login',json={'username':'reader','password':'reader-password-long'})
        assert client.patch('/api/settings/runtime',json={'power_profile':'performance'}).status_code==403
    finally:
        settings.power_profile,settings.runtime_budget_scale=original


def test_prepared_pngs_share_storage_and_preserve_pixels(tmp_path,monkeypatch):
    import numpy as np
    import cv2
    from surveilx.config import settings
    from training.yolo_pipeline import write_prepared_pixels
    monkeypatch.setattr(settings,'data_dir',tmp_path)
    pixels=np.full((32,32,3),77,np.uint8)
    first,second=tmp_path/'one.png',tmp_path/'two.png'
    write_prepared_pixels(first,pixels)
    write_prepared_pixels(second,pixels)
    assert first.read_bytes()==second.read_bytes()
    assert np.array_equal(cv2.imread(str(first)),pixels)
    assert first.stat().st_ino==second.stat().st_ino
