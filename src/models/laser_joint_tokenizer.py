"""Frozen finite joint codeword tables for native RQTransformer.

Every residual depth predicts one ordinary integer indexing a fixed physical
dictionary atom and coefficient. A reserved zero vector is optional.
The tables may allocate different numbers of coefficients to different atoms.
Encoding subtracts each chosen codeword once and never refits an earlier one.
"""
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn


class JointAtomCoefficientRQ(nn.Module):
    def __init__(self, dictionary, atom_ids, coefficients, zero_code_id=0, encoding_beam_width=1):
        super().__init__()
        dictionary = torch.as_tensor(dictionary).detach().float().clone()
        atom_ids = torch.as_tensor(atom_ids).detach().long().clone()
        coefficients = torch.as_tensor(coefficients).detach().float().clone()
        if (dictionary.ndim != 2 or atom_ids.ndim != 2 or
                coefficients.shape != atom_ids.shape or atom_ids.shape[1] < 2):
            raise ValueError('Expected dictionary [C,A] and equal [D,V] ID/value tables')
        if not torch.isfinite(dictionary).all() or not torch.isfinite(coefficients).all():
            raise ValueError('Codebook must be finite')
        if (dictionary.square().sum(0) <= 0).any():
            raise ValueError('Dictionary contains zero atoms')
        if (atom_ids < 0).any() or (atom_ids >= dictionary.shape[1]).any():
            raise ValueError('Atom ID outside dictionary')
        if zero_code_id is None:
            if (coefficients == 0).any():
                raise ValueError('An all-product vocabulary requires nonzero fixed coefficients')
        elif zero_code_id == 0:
            if (coefficients[:, 0] != 0).any() or (coefficients[:, 1:] == 0).any():
                raise ValueError('Reserved ID zero must be the unique zero codeword at each depth')
        else:
            raise ValueError('Only an optional zero code at index0 is supported')
        self.zero_code_id = zero_code_id
        if not isinstance(encoding_beam_width,int) or encoding_beam_width<1:
            raise ValueError('Encoding beam width must be a positive integer')
        self.encoding_beam_width = encoding_beam_width
        self.training_target_mode = 'hard_integer' if encoding_beam_width>1 else 'stochastic_soft'
        self.register_buffer('dictionary', dictionary)
        self.register_buffer('atom_ids', atom_ids)
        self.register_buffer('coefficients', coefficients)
        self.register_buffer('norms', dictionary.square().sum(0))
        self.depth, self.vocab_size = atom_ids.shape

    def embed_depth(self, codes, depth):
        return (self.dictionary.T[self.atom_ids[depth, codes]] *
                self.coefficients[depth, codes, None])

    def embed(self, codes):
        if codes.dtype != torch.long or codes.shape[-1] != self.depth:
            raise ValueError('Expected integer codes with the complete depth axis')
        return torch.stack([self.embed_depth(codes[..., d], d) for d in range(self.depth)], -2)

    def get_code_emb_with_depth(self, codes):
        return self.embed(codes), None

    def expanded_codebook(self, depth):
        ids = torch.arange(self.vocab_size, device=self.dictionary.device)
        return self.embed_depth(ids, depth)

    def scores(self, residual, depth):
        """Exact negative squared distance, omitting the row-constant ||r||²."""
        correlations = residual @ self.dictionary
        atoms, values = self.atom_ids[depth], self.coefficients[depth]
        return 2 * correlations[:, atoms] * values - self.norms[atoms] * values.square()

    @torch.no_grad()
    def quantize(self, x, *, chunk_size=256, return_projections=False):
        if self.encoding_beam_width==1:
            return self._quantize_greedy(x,chunk_size=chunk_size,return_projections=return_projections)
        from src.models.laser_joint_search import beam_quantize
        result=beam_quantize(self,x,beam_width=self.encoding_beam_width,chunk_size=min(chunk_size,256))
        if return_projections:
            residual=x.float().clone();projections=[]
            for depth in range(self.depth):
                ids=result['codes'][...,depth];atoms=self.atom_ids[depth,ids]
                projections.append((residual*self.dictionary.T[atoms]).sum(-1)/self.norms[atoms])
                residual.sub_(self.embed_depth(ids,depth))
            result['projections']=torch.stack(projections,-1)
        return result

    @torch.no_grad()
    def _quantize_greedy(self, x, *, chunk_size=256, return_projections=False):
        residual = x.reshape(-1, x.shape[-1]).float().clone()
        codes = torch.empty(len(residual), self.depth, device=x.device, dtype=torch.long)
        projections = torch.empty_like(codes, dtype=torch.float32) if return_projections else None
        for d in range(self.depth):
            for first in range(0, len(residual), chunk_size):
                r = residual[first:first + chunk_size]
                chosen = self.scores(r, d).argmax(-1)
                if return_projections:
                    atoms = self.atom_ids[d, chosen]
                    projections[first:first + len(r), d] = (
                        (r * self.dictionary.T[atoms]).sum(-1) / self.norms[atoms])
                codes[first:first + len(r), d] = chosen
                r.sub_(self.embed_depth(chosen, d))
        shape = (*x.shape[:-1], self.depth)
        decoded_codes=codes.reshape(shape)
        # Match decode_code exactly. Subtracting the final residual is only
        # algebraically equivalent; its rounding can be amplified by the CNN.
        result = dict(codes=decoded_codes, quantized=self.embed(decoded_codes).sum(-2))
        if return_projections:
            result['projections'] = projections.reshape(shape)
        return result

    @torch.no_grad()
    def get_soft_codes(self, x, temp, stochastic=True, chunk_size=128):
        try:
            temperatures = [float(temp)] * self.depth
        except (TypeError, ValueError):
            temperatures = [float(t) for t in temp]
        if (len(temperatures) != self.depth or
                any(not math.isfinite(t) or t <= 0 for t in temperatures) or chunk_size < 1):
            raise ValueError('Positive finite temperature and chunk size required')
        if self.encoding_beam_width>1:
            # Lookahead chooses an entire fixed integer sequence. A one-hot
            # teacher preserves those exact choices under ordinary AR CE;
            # greedy stochastic soft targets would describe another encoder.
            codes=self.quantize(x,chunk_size=chunk_size)['codes']
            targets=x.new_zeros(*codes.shape,self.vocab_size,dtype=torch.float32)
            targets.scatter_(-1,codes[...,None],1.)
            return targets,codes
        residual = x.reshape(-1, x.shape[-1]).float().clone()
        targets = residual.new_empty(len(residual), self.depth, self.vocab_size)
        codes = torch.empty(len(residual), self.depth, device=x.device, dtype=torch.long)
        for d in range(self.depth):
            for first in range(0, len(residual), chunk_size):
                r = residual[first:first + chunk_size]
                probability = (self.scores(r, d) / temperatures[d]).softmax(-1)
                chosen = (torch.multinomial(probability, 1).squeeze(-1)
                          if stochastic else probability.argmax(-1))
                targets[first:first + len(r), d] = probability
                codes[first:first + len(r), d] = chosen
                r.sub_(self.embed_depth(chosen, d))
        return (targets.reshape(*x.shape[:-1], self.depth, self.vocab_size),
                codes.reshape(*x.shape[:-1], self.depth))


class FrozenJointLaserTokenizer(nn.Module):
    def __init__(self, checkpoint, codebook):
        super().__init__()
        from src.models.laser_tokenizer_adapter import FrozenLaserTokenizer
        spec = torch.load(codebook, map_location='cpu', weights_only=True)
        # Keep recovered research artifacts byte-identical to their W&B hashes.
        # A small sidecar supplies the runtime schema and verified CNN lineage.
        if 'kind' not in spec and 'vectors' in spec:
            metadata = json.loads(Path(codebook).with_suffix('.json').read_text())
            if metadata.get('kind') != 'laser_joint_vector_rq':
                raise ValueError('Expected a joint vector runtime manifest')
            with Path(codebook).open('rb') as handle:
                book_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
            if book_hash != metadata.get('codebook_sha256'):
                raise ValueError('Joint vector codebook SHA256 mismatch')
            spec = dict(metadata, vectors=spec['vectors'])
        merged_book = spec['kind'] == 'laser_joint_merged_vector_rq'
        vector_book = spec['kind'] == 'laser_joint_vector_rq' or merged_book
        if vector_book and 'vector_shards' in spec:
            tables = []
            for shard in spec['vector_shards']:
                path = Path(codebook).parent / shard['name']
                with path.open('rb') as handle:
                    digest = hashlib.file_digest(handle, 'sha256').hexdigest()
                if digest != shard['sha256']:
                    raise ValueError('Joint vector shard SHA256 mismatch')
                tables.append(torch.load(path, map_location='cpu', weights_only=True))
            spec = dict(spec, vectors=torch.stack(tables))
        if not vector_book and spec['kind'] != 'laser_joint_atom_coefficient_rq':
            raise ValueError('Expected a finite joint LASER codebook')
        depth = spec['vectors'].shape[0] if vector_book else spec['atom_ids'].shape[0]
        learned = spec.get('learned_compact_dictionary', False)
        source_atoms = None if vector_book else spec['dictionary'].shape[1]
        if learned or vector_book:
            if (spec.get('format_version') not in ((1,) if vector_book else (2, 3)) or
                    spec.get('encoder_decoder_frozen') is not True):
                raise ValueError('Learned joint codebooks require a versioned frozen-CNN schema')
            with Path(checkpoint).open('rb') as handle:
                source_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
            if source_hash != spec.get('source_stage1_checkpoint_sha256'):
                raise ValueError('Compact dictionary source checkpoint SHA256 mismatch')
            # Keep the original CNN construction (including its RNG consumption)
            # identical to the source tokenizer; only the residual book changes.
            source = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)['state_dict']
            key = 'quantizer.dictionary' if 'quantizer.dictionary' in source else 'bottleneck.dictionary'
            source_atoms = source[key].shape[1]
            if vector_book and spec['vectors'].shape[-1] != source[key].shape[0]:
                raise ValueError('Joint vector channels differ from the source CNN')
            del source
        self.backbone = FrozenLaserTokenizer(checkpoint, num_atoms=source_atoms,
            coeff_vocab_size=2, coeff_max=1., coeff_scales=[1.] * depth, sparsity_level=depth)
        if not learned and not vector_book:
            torch.testing.assert_close(spec['dictionary'], self.backbone.dictionary, rtol=0, atol=0)
        if merged_book:
            from src.models.joint_vocab_merge import MergedJointVectorBook, load_encoder_vectors
            self.quantizer = MergedJointVectorBook(spec['vectors'],
                load_encoder_vectors(spec, Path(codebook).parent), spec['remap'],
                coordinate_sweeps=int(spec.get('coordinate_sweeps', 0)))
        elif vector_book:
            from src.models.joint_rate_distortion import JointVectorBook
            self.quantizer = JointVectorBook(spec['vectors'])
        else:
            self.quantizer = JointAtomCoefficientRQ(spec['dictionary'], spec['atom_ids'], spec['coefficients'],
                zero_code_id=spec.get('zero_code_id', 0),
                encoding_beam_width=spec.get('encoding_beam_width', 1))
        if spec.get('shared_codebook_across_depths', False):
            if vector_book:
                if not torch.equal(spec['vectors'], spec['vectors'][:1].expand_as(spec['vectors'])):
                    raise ValueError('Joint vector tables are not shared across depths')
            else:
                from src.training.laser_shared_joint_vocabulary import assert_shared_codewords
                assert_shared_codewords(self.quantizer)
        self.code_shape = torch.Size([8, 8, depth])
        self.requires_grad_(False).eval()

    def encode(self, images):
        return self.backbone.quant_conv(self.backbone.encoder(images)).permute(0, 2, 3, 1).contiguous()

    def decode(self, latent):
        return self.backbone.decoder(self.backbone.post_quant_conv(latent.permute(0, 3, 1, 2).contiguous()))

    def decode_code(self, codes):
        return self.decode(self.quantizer.embed(codes).sum(-2))

    def get_code_emb_with_depth(self, codes):
        return self.quantizer.get_code_emb_with_depth(codes)

    def forward(self, images):
        latent = self.encode(images)
        result = self.quantizer.quantize(latent)
        return self.decode(result['quantized']), latent.new_zeros(()), result['codes']
