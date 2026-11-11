"""Install semantics-preserving startup and first-batch integrity checks."""
from pathlib import Path
import difflib
import shutil

ROOT=Path('/workspace/Projects/laser')
BASE=Path('/tmp/laser-beam451-stage2-4h100-20261107')
AUDIT=ROOT/'outputs/imagenet-beam451-aligned-joint16k-scratch100-4h100-20261107/audit-stage1-preservation-20261108'


def main():
    driver=BASE/'source/src/training/rq_imagenet_shared.py'
    before=driver.read_text();assert 'preservation_metadata' not in before
    old='    aux,stage1_config=PROFILE.tokenizer(device)'
    assert before.count(old)==1
    code=before.replace(old,old+'\n    from runtime.preservation_guard import verify_frozen_codec\n    preservation_metadata=verify_frozen_codec(PROFILE,aux)')
    old='**PROFILE.metadata(),**rng_metadata)';assert code.count(old)==1
    code=code.replace(old,'**PROFILE.metadata(),**rng_metadata,**preservation_metadata)')
    compile(code,str(driver),'exec');driver.write_text(code)
    recipe=BASE/'runtime/aligned_recipe.py'
    old_recipe=recipe.read_text()
    old='            return aligned_targets(aux.physical_book, latent, self.target_temperature())'
    assert old_recipe.count(old)==1
    new='''            if not hasattr(self,'target_preservation_audit'):
                from runtime.preservation_guard import verify_teacher
                targets,codes,pool,weights,chosen=aligned_targets(aux.physical_book,latent,
                    self.target_temperature(),return_details=True)
                self.target_preservation_audit=verify_teacher(pool,weights,chosen)
                return targets,codes
            return aligned_targets(aux.physical_book, latent, self.target_temperature())'''
    new_recipe=old_recipe.replace(old,new)
    old='        return result\n\n    def metadata(self):';assert new_recipe.count(old)==1
    new_recipe=new_recipe.replace(old,"        result.update(self.target_preservation_audit)\n"+old)
    compile(new_recipe,str(recipe),'exec');recipe.write_text(new_recipe)
    shutil.copyfile(Path(__file__).with_name('preservation_guard.py'),BASE/'runtime/preservation_guard.py')
    patch=''.join(difflib.unified_diff(before.splitlines(True),code.splitlines(True),fromfile='rng-fixed/rq_imagenet_shared.py',tofile='guarded/rq_imagenet_shared.py'))
    patch+=''.join(difflib.unified_diff(old_recipe.splitlines(True),new_recipe.splitlines(True),fromfile='original/aligned_recipe.py',tofile='guarded/aligned_recipe.py'))
    (AUDIT/'preservation-guards.patch').write_text(patch)
    print('Installed integrity guards for next checkpoint resume')


if __name__=='__main__':main()
