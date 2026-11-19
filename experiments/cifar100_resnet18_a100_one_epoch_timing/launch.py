def launch_timing(method):
    assert method in ('SGD','SAM','IGR')
    assert hashlib.sha256(SCRIPT.read_bytes()).hexdigest()==MANIFEST['worker_sha256']
    assert hashlib.sha256((SESSION/'config.json').read_bytes()).hexdigest()==MANIFEST['config_sha256']
    marker=SESSION/(method+'_launch.json')
    assert not marker.exists(), 'This attempt is preserved. Rerun setup for a fresh timing folder.'
    atomic_json(marker,dict(status='running',manifest=MANIFEST))
    start=time.monotonic();status='failed'
    try:
        command([PYTHON,'-u',SCRIPT,SESSION/'config.json',method],SESSION/(method+'.log'),ENV,DEADLINE)
        assert json.loads((SESSION/(method+'_status.json')).read_text())['status']=='complete'
        result=json.loads((SESSION/(method+'_result.json')).read_text())
        assert result['epochs_measured']==1 and result['steps']==351 and result['status']=='complete'
        status='complete'
    finally:atomic_json(marker,dict(status=status,manifest=MANIFEST,wall_seconds=time.monotonic()-start))

