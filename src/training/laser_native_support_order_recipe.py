"""Native eight-event RQ with unchanged compound512 target extraction."""
from src.training.laser_compound_replay_recipe import CompoundReplayRecipe
from src.training.rq_imagenet_recipes import LaserRecipe


class NativeSupportOrderRecipe(CompoundReplayRecipe):
    kind='laser_native_support_order'

    def __init__(self,base):
        LaserRecipe.__init__(self,base)
        assert self.recipe.decoder=='native_support_order'
        assert self.recipe.event_order in ('interleaved','support_first')
        assert self.recipe.tokenizer.num_atoms==16384 and self.recipe.tokenizer.coefficient_bins==512
        assert self.recipe.targets.temperature==.03125 and self.recipe.atom_ce_weight==1.

    def model(self,arch):
        from src.models.laser_native_support_order_adapter import build_native_support_order_model
        return build_native_support_order_model(arch,num_atoms=16384,coefficient_bins=512,
            order=str(self.recipe.event_order))

    def metadata(self):
        result=super().metadata()
        result.update(architecture='native RQ spatial and eight-event depth transformers with typed output slices',
            objective='joint categorical negative log likelihood averaged over eight events',
            atom_ce_weight=1.,coefficient_ce_weight=1.,
            event_order=str(self.recipe.event_order),native_depth_events_per_site=8,
            separate_coefficient_micro_transformer=False,codec_changed=False,
            coefficient_conditioning='all four current supports' if self.recipe.event_order=='support_first'
                else 'previous completed pairs and selected current support',
            atom_conditioning='previous support IDs, without current-site coefficient noise'
                if self.recipe.event_order=='support_first' else 'previous completed pairs')
        return result
