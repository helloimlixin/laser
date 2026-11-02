"""Publish run metadata without flushing a concurrent writer's summary."""


def update_config_only(run, updates):
    """Avoid PublicRun.update(), which also writes its cached summary snapshot."""
    run.config.update(updates)
    return run._exec('''mutation UpdateRunConfigOnly($id:String!,$config:JSONString!){
        upsertBucket(input:{id:$id,config:$config}){bucket{id}}}''',
        id=run.storage_id, config=run.json_config)
