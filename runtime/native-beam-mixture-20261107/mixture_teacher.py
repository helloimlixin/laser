"""Full-support native/beam sequence mixture with exact prefix conditionals.

For every latent site choose one branch for all four depths:
q(c|x) = a p_native(c|x,T=.5) + (1-a) 1[c=beam_best(x)].
Labels use the posterior branch weights given the actual sampled prefix.
There is no near-best error mask and no finite-beam support restriction on
the native component. Neither physical codewords nor CNN weights change.
"""
import math
import torch


def conditional(native_probability,log_native_prefix,beam_compatible,beam_token,native_weight):
    a=float(native_weight)
    if not 0<=a<=1:raise ValueError('Native weight must be in [0,1]')
    if a==1:return native_probability
    if a==0:
        target=torch.zeros_like(native_probability)
        return target.scatter_(-1,beam_token[...,None],1.)
    log_native=log_native_prefix+math.log(a)
    log_beam=torch.full_like(log_native,math.log1p(-a)).masked_fill(~beam_compatible,-torch.inf)
    normalizer=torch.logaddexp(log_native,log_beam)
    native_posterior=(log_native-normalizer).exp()
    beam_posterior=(log_beam-normalizer).exp()
    target=native_probability*native_posterior[...,None]
    target.scatter_add_(-1,beam_token[...,None],beam_posterior[...,None])
    return target


@torch.no_grad()
def canonical_beam_codes(aux,latent):
    from runtime import beam_teacher
    pool=beam_teacher.beam_pool(aux.physical_book,latent)
    row=torch.arange(len(pool['costs']),device=latent.device)
    return pool['candidates'][row,pool['canonical_index']].reshape(*latent.shape[:-1],4)


@torch.no_grad()
def get_targets(aux,latent,native_weight,*,temperature=.5,beam_codes=None,return_details=False):
    a=float(native_weight)
    if not 0<=a<=1:raise ValueError('Native weight must be in [0,1]')
    if a==1:
        targets,codes=aux.quantizer.get_soft_codes(latent,temp=temperature,stochastic=True)
        if return_details:return targets,codes,dict(native_branch=torch.ones_like(codes[...,0],dtype=torch.bool))
        return targets,codes
    if beam_codes is None:beam_codes=canonical_beam_codes(aux,latent)
    if a==0:
        target=latent.new_zeros(*beam_codes.shape,aux.quantizer.vocab_size)
        target.scatter_(-1,beam_codes[...,None],1.)
        if return_details:return target,beam_codes,dict(native_branch=torch.zeros_like(beam_codes[...,0],dtype=torch.bool))
        return target,beam_codes
    residual=aux.quantizer.to_code_shape(latent).detach().clone()
    shape=residual.shape[:-1]
    native_branch=torch.rand(shape,device=latent.device)<a
    log_native_prefix=torch.zeros(shape,device=latent.device,dtype=torch.float32)
    compatible=torch.ones(shape,device=latent.device,dtype=torch.bool)
    probability_list=[];codes=[];joint_probability=torch.ones_like(log_native_prefix)
    for depth,book in enumerate(aux.quantizer.codebooks):
        probability=(-book.compute_distances(residual)/float(temperature)).softmax(-1)
        target=conditional(probability,log_native_prefix,compatible,beam_codes[...,depth],a)
        sampled=torch.multinomial(probability.reshape(-1,probability.shape[-1]),1).reshape(shape)
        code=torch.where(native_branch,sampled,beam_codes[...,depth])
        native_selected=probability.gather(-1,code[...,None]).squeeze(-1)
        if return_details:
            joint_probability*=target.gather(-1,code[...,None]).squeeze(-1)
        log_native_prefix+=native_selected.log()
        compatible &= code==beam_codes[...,depth]
        residual-=book.embed(code)
        probability_list.append(target.unsqueeze(-2));codes.append(code[...,None])
    targets=torch.cat(probability_list,-2);codes=torch.cat(codes,-1)
    if return_details:
        expected=a*log_native_prefix.exp()+(1-a)*compatible
        return targets,codes,dict(native_branch=native_branch,
            joint_probability=joint_probability,expected_joint_probability=expected,
            log_native_prefix=log_native_prefix)
    return targets,codes


def verify_bayes():
    import itertools
    beam=(1,0,2);vocab=3;depth=3
    def native(prefix):
        raw=torch.tensor([1.+sum(prefix),2.+len(prefix),4.+(prefix[-1] if prefix else 0)],dtype=torch.float64)
        return raw/raw.sum()
    for a in [0.,.1,.25,.5,1.]:
        total=0.
        for sequence in itertools.product(range(vocab),repeat=depth):
            native_joint=1.;chain=1.;log_prefix=torch.zeros(1,dtype=torch.float64);match=True
            for d,token in enumerate(sequence):
                p=native(sequence[:d])[None]
                mixed=conditional(p,log_prefix,torch.tensor([match]),torch.tensor([beam[d]]),a)
                torch.testing.assert_close(mixed.sum(-1),torch.ones(1,dtype=torch.float64),atol=1e-13,rtol=1e-13)
                chain*=float(mixed[0,token]);native_joint*=float(p[0,token])
                log_prefix+=p[0,token].log();match &= token==beam[d]
            expected=a*native_joint+(1-a)*(sequence==beam)
            assert abs(chain-expected)<1e-12,(a,sequence,chain,expected)
            total+=chain
        assert abs(total-1)<1e-12
    return dict(passed=True,all_27_sequences_checked=True,weights=[0.,.1,.25,.5,1.],
                exact_joint_probability_from_prefix_conditionals=True)
