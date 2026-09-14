"""Fit frozen atom-specific coefficients to the final residual reconstruction."""
import torch


@torch.no_grad()
def coefficient_normal_equations(dictionary, codes, targets, levels_per_atom):
    """Return A.T A and A.T z for a fixed sparse four-code reconstruction.

    A row block sums dictionary columns times their selected coefficient.
    Repeated codes accumulate into the same column; zero has no coefficient.
    """
    codes = codes.reshape(-1, codes.shape[-1])
    targets = targets.reshape(-1, dictionary.shape[0])
    packed = (codes-1).clamp_min(0)
    vectors = dictionary.T[packed//levels_per_atom].double()*(codes!=0)[...,None]
    products = vectors@vectors.transpose(-1,-2)
    rows = packed[:,:,None].expand_as(products)
    cols = packed[:,None,:].expand_as(products)
    active = (codes[:,:,None]!=0)&(codes[:,None,:]!=0)
    size = dictionary.shape[1]*levels_per_atom
    normal = torch.sparse_coo_tensor(torch.stack([rows[active],cols[active]]),products[active],
        (size,size),device=dictionary.device)
    rhs = torch.zeros(size,device=dictionary.device,dtype=torch.float64)
    rhs.index_add_(0,packed.flatten(),(vectors*targets.double()[:,None]).sum(-1).flatten())
    return normal, rhs


@torch.no_grad()
def solve_regularized_levels(normal, rhs, current, prior, ridge, iterations=100, tolerance=1e-9):
    """Preconditioned conjugate gradients with a positive diagonal prior."""
    normal = normal.coalesce()
    ridge = torch.broadcast_to(torch.as_tensor(ridge,device=rhs.device,dtype=rhs.dtype),rhs.shape)
    if not (ridge>0).all():
        raise ValueError('Positive coefficient regularization required')
    def product(x):
        return torch.sparse.mm(normal,x[:,None]).squeeze(1)+ridge*x
    b = rhs+ridge*prior.flatten().double()
    diagonal = ridge.clone()
    ij = normal.indices()
    mask = ij[0]==ij[1]
    diagonal.index_add_(0,ij[0,mask],normal.values()[mask])
    x = current.flatten().double().clone()
    residual = b-product(x)
    z = residual/diagonal
    direction = z.clone()
    rz = residual@z
    target = tolerance*b.norm().clamp_min(1.)
    for _ in range(iterations):
        if residual.norm()<=target:
            break
        projected = product(direction)
        denominator = direction@projected
        if denominator<=0 or not torch.isfinite(denominator):
            raise FloatingPointError('Coefficient system is not positive definite')
        alpha = rz/denominator
        x += alpha*direction
        residual -= alpha*projected
        z = residual/diagonal
        new_rz = residual@z
        direction = z+(new_rz/rz)*direction
        rz = new_rz
    if not torch.isfinite(x).all():
        raise FloatingPointError('Nonfinite fitted coefficient')
    return x.reshape_as(current).to(current.dtype)
