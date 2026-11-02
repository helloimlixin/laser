"""Restrict sampler-only continuation to a measured, one-setting improvement."""
import copy


def validated_sampling_recipe(current, saved, evidence):
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    require(evidence.get('passed') is True, 'Sampler validation did not pass')
    require(evidence.get('weights_unmodified') is True and evidence.get('training_updates_added') == 0,
        'Sampler evidence must use fixed weights')
    require(evidence.get('generated_images') == 50000 and evidence.get('reference_images') == 1281167
        and evidence.get('reference_split') == 'train', 'Sampler requires the full FID50k protocol')
    require(evidence.get('source_run') == current['run_id'], 'Sampler evidence belongs to another run')
    require(evidence.get('fid',float('inf')) < evidence.get('baseline_fid',float('-inf')),
        'Candidate sampler did not improve matched FID')
    source, target = evidence['source_sampling'], evidence['sampling']
    require(set(source) == set(target), 'Sampler vocabulary/settings differ')
    changed = [key for key in source if source[key] != target[key]]
    legacy_atom_change = (changed == ['atom_top_k']
        and source['atom_top_k'] == 256 and target['atom_top_k'] == 0)
    joint_integer_change = (set(source) == {'temperature', 'top_k', 'top_p'}
        and changed == ['top_k'] and source['top_k'] == 512 and target['top_k'] == 2048)
    require(legacy_atom_change or joint_integer_change,
        'Only validated atom-top-k removal or joint-integer K512 to K2048 is permitted')
    require(current['sampling'] in (source,target) and saved['sampling'] in (source,target),
        'Checkpoint or configured sampler differs from validated settings')
    current_training = {k:v for k,v in current.items() if k != 'sampling'}
    saved_training = {k:v for k,v in saved.items() if k != 'sampling'}
    require(current_training == saved_training, 'Sampler override cannot change the training recipe')
    result = copy.deepcopy(saved)
    result['sampling'] = copy.deepcopy(target)
    return result
