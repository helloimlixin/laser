"""Local held-out likelihood probe; never reports a substitute image metric."""
import torch


@torch.no_grad()
def measure(model, batch_file, stage1_checkpoint, args, device):
    from src.training.physical_pair_crps import physical_pair_objective_components
    # The frozen encoder is unnecessary here: use only its tiny dictionary
    # and coefficient buffers, already attached to the main training aux.
    aux=model._compound_probe_aux
    batch=torch.load(batch_file,map_location='cpu',weights_only=False)
    was_training=model.training
    values=[]
    with torch.random.fork_rng(devices=[device.index]):
        model.eval()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            for token,(atoms,probabilities),label in zip(batch['tokens'],batch['targets'],batch['labels']):
                token=token.to(device);atoms=atoms.to(device);probabilities=probabilities.to(device)
                output=model(token,aux,torch.tensor([label],device=device))
                loss,ce,crps=physical_pair_objective_components(output['atom_logits'],output['coeff_logits'],
                    atoms,probabilities,aux.coeff_bins,torch.tensor(.05,device=device),1)
                values.append(float(ce))
        model.train(was_training)
    result=dict(mean_cross_entropy=sum(values)/len(values),images=len(values),values=values,
        baseline_cpu_cross_entropy=6.659251054128011,finite=all(torch.isfinite(torch.tensor(v)) for v in values),
        history_gate_mean=float(model.history_gates.float().mean()),diagnostic_only=True)
    return result
