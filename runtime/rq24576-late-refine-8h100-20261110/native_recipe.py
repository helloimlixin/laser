"""Late continuation: frozen 24,576 words, annealed noise, explicit EMA state."""
import json
import os
import torch
from omegaconf import OmegaConf
from runtime.native_parent_recipe import NativeRecipe as ParentRecipe
from runtime.refinement import teacher_temperature


class NativeRecipe(ParentRecipe):
    def __init__(self,base):
        super().__init__(base)
        self.refinement=json.loads((self.base/'inputs/late-refinement.json').read_text())
        self.parent_run_id=self.refinement['parent_run_id']
        self.training_step=self.refinement['source_step']

    def set_training_step(self,step):self.training_step=int(step)

    def target_temperature(self):
        if os.environ.get('LASER_TEST_FINAL_TEACHER')=='1':
            return self.refinement['teacher']['final']
        return teacher_temperature(self.training_step,self.refinement['teacher'])

    def sampling_settings(self):return dict(self.refinement['sampler'])

    def logging_sampling_settings(self):
        return {'sampler_'+key:value for key,value in self.sampling_settings().items()}

    @torch.no_grad()
    def sample(self,model,aux,count,labels):
        settings=self.sampling_settings();temperature=settings.pop('temperature')
        partial=torch.zeros(count,*self.shape,dtype=torch.long,device=labels.device)
        if not isinstance(temperature,list):
            return model.sample(partial,model_aux=aux,cond=labels,amp=True,temperature=temperature,**settings)
        from rqvae.models.rqtransformer import transformers as module
        original=module.sample_from_logits;calls=0
        def draw(logits,*args,**kwargs):
            nonlocal calls
            kwargs['temperature']=float(temperature[calls%4]);calls+=1
            return original(logits,*args,**kwargs)
        module.sample_from_logits=draw
        try:result=model.sample(partial,model_aux=aux,cond=labels,amp=True,temperature=1.,**settings)
        finally:module.sample_from_logits=original
        assert calls==256
        return result

    def verify_resume(self,payload):
        if payload['run_id']==self.parent_run_id:
            parent=OmegaConf.to_container(OmegaConf.load(self.base/'inputs/parent-recipe.yaml'),resolve=True)
            assert payload['recipe']==parent
            assert payload['global_step']==self.refinement['source_step']
            assert payload['next_epoch']==self.refinement['source_epoch'] and payload['next_microbatch']==0
            assert 'late_refinement' not in payload['scheduler']
        else:
            assert payload['run_id']==self.run_id
            assert payload['recipe']==OmegaConf.to_container(self.recipe,resolve=True)
            assert payload['scheduler']['late_refinement']==self.refinement['lr']
            assert payload['state_dict_weights']=='ema' and payload['optimizer_weights_key']=='state_dict_raw'
            assert payload['ema_decay']==self.refinement['ema_decay']
            assert payload['ema_updates']==payload['global_step']-self.refinement['source_step']
            assert set(payload['state_dict'])==set(payload['state_dict_raw'])
        assert payload['joint_codebook_sha256']==self.codebook_sha256

    def checkpoint_extra(self):
        return dict(super().checkpoint_extra(),late_refinement=self.refinement,
            state_dict_weights='ema',optimizer_weights_key='state_dict_raw',
            ema_decay=self.refinement['ema_decay'],evaluation_weights='ema')

    def metadata(self):
        result=super().metadata()
        result.update(late_refinement=self.refinement,parent_run_id=self.parent_run_id,
            source_epoch=self.refinement['source_epoch'],source_checkpoint_sha256=self.refinement['source_checkpoint_sha256'],
            original_optimizer_moments_retained=True,original_optimizer_age_retained=True,
            teacher_temperature_schedule=self.refinement['teacher'],ema_decay=self.refinement['ema_decay'],
            fid_target=11.56,evaluation_weights='ema')
        result['initialization']=dict(result['initialization'],stage2_checkpoint_loaded=True,
            source_epoch=self.refinement['source_epoch'])
        return result
