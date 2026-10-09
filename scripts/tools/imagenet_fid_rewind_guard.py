"""Stop at an official evaluated checkpoint before any further Adam update."""
import math
from pathlib import Path
import shutil


def regression_request(metrics, best_fid, decision):
    if (metrics.get('metric_backend') != 'original_rqtransformer'
            or metrics.get('real_images') != 50000
            or metrics.get('generated_images') != 50000):
        raise ValueError('Automatic rewind requires the official 50k evaluation')
    if not best_fid:
        raise ValueError('A protected full best-FID checkpoint is required')
    fid = float(metrics['fid'])
    best, source = min(best_fid, key=lambda item: item[0])
    if not math.isfinite(fid) or not math.isfinite(float(best)):
        raise ValueError('FID must be finite')
    if fid <= float(best):
        return None
    if decision is None or not math.isfinite(decision['lr_after']) or decision['lr_after'] < 0:
        raise ValueError('A valid saved LR decision is required')
    return dict(reason='official FID worse than saved best',
                global_step=metrics['global_step'], fid=fid, best_fid=float(best),
                source_checkpoint=str(source), requested_lr=decision['lr_after'],
                metric_backend='original_rqtransformer', strict_comparison=True)


def install_guard(base):
    """Patch a frozen runtime once; native epoch saves still finish normally."""
    base = Path(base)
    for name in ('imagenet_fid_rewind_guard.py', 'imagenet_fid_rewind_supervisor.py',
                 'imagenet_epoch64_lower_lr.py', 'imagenet_zero_floor_lr.py',
                 'imagenet_lower_floor_lr.py', 'continue_imagenet_epoch64_lower_lr.py',
                 'continue_imagenet_fid16379.py', 'continue_imagenet_pairfix_repair.py',
                 'drain_imagenet_full_checkpoints.py', 'continue_imagenet_coeff_crps.py',
                 'continue_imagenet_lr1000.py', 'imagenet_scale_lr.py'):
        source = Path(__file__).with_name(name)
        target = base / 'support' / name
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)
    trainer = base / 'source/runtime/src/training/rqtransformer.py'
    code = trainer.read_text()
    marker = "        # Stop at the official scored checkpoint, before the next epoch.\n"
    if marker not in code:
        old = ('        if dist.is_initialized():\n'
               '            dist.barrier()\n'
               '    if wb is not None:\n'
               '        if hasattr(upload_selected_checkpoint_files, "close"):\n')
        new = ('        if dist.is_initialized():\n'
               '            dist.barrier()\n' + marker +
               '        if getattr(args, "stop_after_fid_regression", False):\n'
               '            break\n'
               '    if wb is not None:\n'
               '        if hasattr(upload_selected_checkpoint_files, "close"):\n')
        assert code.count(old) == 1
        trainer.write_text(code.replace(old, new))
    entry = base / 'entry.py'
    code = entry.read_text()
    if 'FID_REWIND_REQUEST=None' in code:
        return
    def replace(old, new):
        nonlocal code
        assert code.count(old) == 1, old
        code = code.replace(old, new)
    replace('STOPPING=False\n', 'STOPPING=False\nFID_REWIND_REQUEST=None\n')
    replace(' global OFFICIAL_METRICS\n', ' global OFFICIAL_METRICS,STOPPING,FID_REWIND_REQUEST\n')
    replace(' config.update(resume_source_epoch=',
            " config.update(automatic_fid_rewind=True,fid_rewind_policy='strictly worse than the saved best; no further updates; full-state rewind at reduced LR')\n config.update(resume_source_epoch=")
    replace(' return result\ntraining.evaluate_generation_metrics=evaluate\n',
            " from imagenet_fid_rewind_guard import regression_request\n"
            " request=[regression_request(OFFICIAL_METRICS,BEST_FID,decision) if dist.get_rank()==0 else None]\n"
            " dist.broadcast_object_list(request,src=0)\n"
            " if request[0] is not None:\n"
            "  STOPPING=True;ARGS.stop_after_fid_regression=True;FID_REWIND_REQUEST=request[0]\n"
            "  record(VERIFY/('fid-rewind-rank'+os.environ['RANK']+'.json'),dict(request=request[0],updates=UPDATES,adam_step=INITIAL_ADAM_STEP+UPDATES,no_further_updates=True))\n"
            "  if dist.get_rank()==0:\n"
            "   record(EVIDENCE/'fid-rewind-request.json',request[0])\n"
            "   print('Official FID regression: stopping at the evaluated checkpoint for full-state rewind',flush=True)\n"
            " return result\ntraining.evaluate_generation_metrics=evaluate\n")
    replace('  PREVIEW_WRITER.close();WRITER.close();submit_upload()\n',
            "  PREVIEW_WRITER.close();WRITER.close();submit_upload()\n"
            "  if FID_REWIND_REQUEST is not None:\n"
            "   assert INITIAL_RECOVERY['global_step']+UPDATES==FID_REWIND_REQUEST['global_step']\n"
            "   record(EVIDENCE/'fid-rewind-checkpoint-ready.json',dict(request=FID_REWIND_REQUEST,best_fid=BEST_FID,best_is=BEST_IS,updates=UPDATES,no_further_updates=True,time=time.time()))\n")
    entry.write_text(code)
