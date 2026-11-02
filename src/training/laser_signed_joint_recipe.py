"""Native four-step RQ prior with jointly learned atom/sign output classes."""
import torch

from src import rq_upstream
from src.training.laser_fine_joint_recipe import FineJointLaserRecipe
from src.training.rq_imagenet_recipes import RQRecipe


class SignedJointLaserRecipe(FineJointLaserRecipe):
    kind = 'laser_signed_joint'
    decoder = 'signed_atom_joint_integer'

    def model(self, arch):
        from src.models.laser_signed_joint_adapter import install_signed_joint_head
        model = RQRecipe.model(self, arch)
        assert type(model) is rq_upstream.RQTransformer
        assert model.config.input_emb_vqvae and model.config.head_emb_vqvae
        return install_signed_joint_head(model, int(self.recipe.tokenizer.num_atoms),
            int(self.recipe.tokenizer.coefficient_bins), int(self.recipe.head.context_width))

    def loss(self, model, features, targets):
        from src.models.laser_signed_joint_adapter import signed_joint_cross_entropy
        loss, self.last_loss_metrics = signed_joint_cross_entropy(model.classifier.linear, features, targets)
        return loss

    @torch.no_grad()
    def sample(self, model, aux, count, labels):
        from rqvae.utils.utils import sample_from_logits
        from src.models.laser_signed_joint_adapter import merge_coefficient_bins
        s, head = self.recipe.sampling, model.classifier.linear
        codes = torch.zeros(count, *self.shape, dtype=torch.long, device=labels.device)
        model.init_cache()
        try:
            for h in range(8):
                for w in range(8):
                    for d in range(4):
                        features = model.cached_forward(codes[:, :h+1], aux, labels, amp=True,
                                                        sample_loc=(h, w, d))
                        signed_logits = features[:, :head.signed_atom_classes].float().clone()
                        if d:
                            previous = codes[:, h, w, :d]//aux.coeff_vocab_size
                            signed_logits.scatter_(-1, 2*previous, -torch.inf)
                            signed_logits.scatter_(-1, 2*previous+1, -torch.inf)
                        signed_logits = (signed_logits/float(s.atom_temperature)).reshape(count, aux.num_atoms, 2)
                        # Preserve the previous experiment's atom-level top-k
                        # policy. These two draws sample the jointly classified
                        # atom/sign law by its exact marginal and conditional.
                        atoms = sample_from_logits(signed_logits.logsumexp(-1), temperature=1.,
                            top_k=int(s.atom_top_k) or None, top_p=float(s.atom_top_p)).reshape(count)
                        sign_logits = signed_logits.gather(1, atoms[:, None, None].expand(count, 1, 2))[:, 0]
                        signs = sample_from_logits(sign_logits, temperature=1.).reshape(count)
                        signed_atoms = 2*atoms+signs
                        magnitude_logits = head.magnitude_logits(features, signed_atoms)
                        magnitude = sample_from_logits(magnitude_logits.float(),
                            temperature=float(s.coefficient_temperature),
                            top_k=int(s.coefficient_top_k) or None,
                            top_p=float(s.coefficient_top_p)).reshape(count)
                        bins = merge_coefficient_bins(signs, magnitude, aux.coeff_vocab_size)
                        codes[:, h, w, d] = atoms*aux.coeff_vocab_size+bins
        finally:
            model.init_cache()
        return codes

    def metadata(self):
        data = super().metadata()
        data.update(architecture='upstream RQTransformer with joint atom/sign classifier and conditional magnitude head',
            atom_classifier_width=2*int(self.recipe.tokenizer.num_atoms),
            signed_atom_classes=2*int(self.recipe.tokenizer.num_atoms),
            magnitude_bins=int(self.recipe.tokenizer.coefficient_bins)//2,
            joint_probability='p(atom,sign|prefix)*p(magnitude|atom,sign,prefix)',
            objective='joint_categorical_cross_entropy',
            coefficient_soft_labels_preserved_exactly=True,
            atom_truncation='unchanged marginal-atom top-k/top-p; sign from joint classifier conditional',
            loss_units='nats per complete atom/coefficient pair; scalar-event metric divides by two',
            loss_component_diagnostics=True)
        return data
