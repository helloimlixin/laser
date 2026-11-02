"""FP32 batched kernels for LASER's tiny OMP systems and residual products."""
import torch
import triton
import triton.language as tl


@triton.jit
def _extend_cholesky_kernel(matrix, gram, support, atom, output, invalid,
                            rows: tl.constexpr, previous: tl.constexpr,
                            vocabulary: tl.constexpr, block: tl.constexpr):
    row=tl.program_id(0)*block+tl.arange(0,block)
    valid=row<rows
    selected=tl.load(atom+row,valid,0)
    solved=()
    squared=tl.full((block,),0,tl.float32)
    depth: tl.constexpr=previous+1
    for i in tl.static_range(previous):
        existing=tl.load(support+row*previous+i,valid,0)
        value=tl.load(gram+existing*vocabulary+selected,valid,0)
        for j in tl.static_range(i):
            value=value-tl.load(matrix+row*previous*previous+i*previous+j,valid,0)*solved[j]
        value=value/tl.load(matrix+row*previous*previous+i*previous+i,valid,1)
        solved+=(value,)
        squared=squared+value*value
        for j in tl.static_range(previous):
            old=tl.load(matrix+row*previous*previous+i*previous+j,valid,0)
            tl.store(output+row*depth*depth+i*depth+j,old,valid)
        tl.store(output+row*depth*depth+i*depth+previous,0,valid)
        tl.store(output+row*depth*depth+previous*depth+i,value,valid)
    diagonal=tl.load(gram+selected*vocabulary+selected,valid,1)-squared
    tl.store(invalid+row,diagonal<=1.e-7,valid)
    tl.store(output+row*depth*depth+previous*depth+previous,
        tl.sqrt(tl.maximum(diagonal,1.e-7)),valid)


@triton.jit
def _support_probabilities_kernel(residual, available, output, vocabulary: tl.constexpr,
                                  temperature: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0)
    column = tl.arange(0, block)
    valid = column < vocabulary
    value = tl.load(residual + row*vocabulary+column, valid, 0)
    allowed = tl.load(available + row*vocabulary+column, valid, 0)
    score = tl.where(valid & allowed, value*value, -float('inf'))
    shifted = (score-tl.max(score, 0))/temperature
    probabilities = tl.exp(shifted)
    probabilities = probabilities/tl.sum(probabilities, 0)
    tl.store(output + row*vocabulary+column, probabilities, valid)


@triton.jit
def _solve_lower_kernel(matrix, rhs, output, rows: tl.constexpr, depth: tl.constexpr,
                        block: tl.constexpr):
    row = tl.program_id(0)*block + tl.arange(0, block)
    valid = row < rows
    solved = ()
    for i in tl.static_range(depth):
        value = tl.load(rhs + row*depth+i, valid, 0)
        for j in tl.static_range(i):
            value = value - tl.load(matrix + row*depth*depth+i*depth+j, valid, 0)*solved[j]
        value = value/tl.load(matrix + row*depth*depth+i*depth+i, valid, 1)
        solved += (value,)
        tl.store(output + row*depth+i, value, valid)


@triton.jit
def _solve_cholesky_kernel(matrix, rhs, output, rows: tl.constexpr, depth: tl.constexpr,
                           block: tl.constexpr):
    row = tl.program_id(0)*block + tl.arange(0, block)
    valid = row < rows
    forward = ()
    for i in tl.static_range(depth):
        value = tl.load(rhs + row*depth+i, valid, 0)
        for j in tl.static_range(i):
            value = value - tl.load(matrix + row*depth*depth+i*depth+j, valid, 0)*forward[j]
        forward += (value/tl.load(matrix + row*depth*depth+i*depth+i, valid, 1),)
    backward = ()
    for i in tl.static_range(depth-1, -1, -1):
        value = forward[i]
        for j in tl.static_range(i+1, depth):
            value = value - tl.load(matrix + row*depth*depth+j*depth+i, valid, 0)*backward[depth-1-j]
        value = value/tl.load(matrix + row*depth*depth+i*depth+i, valid, 1)
        backward += (value,)
        tl.store(output + row*depth+i, value, valid)


@triton.jit
def _residual_kernel(correlation, coefficients, gram, support, output,
                     count: tl.constexpr, vocabulary: tl.constexpr,
                     depth: tl.constexpr, block: tl.constexpr):
    index = tl.program_id(0)*block + tl.arange(0, block)
    valid = index < count
    row, column = index//vocabulary, index%vocabulary
    product = tl.full((block,), 0, tl.float32)
    for d in tl.static_range(depth):
        atom = tl.load(support + row*depth+d, valid, 0)
        weight = tl.load(coefficients + row*depth+d, valid, 0)
        product = product + weight*tl.load(gram + atom*vocabulary+column, valid, 0)
    value = tl.load(correlation + index, valid, 0)-product
    tl.store(output + index, value, valid)


def solve_lower(matrix, rhs):
    rhs = rhs.contiguous()
    output = torch.empty_like(rhs)
    rows, depth = matrix.shape[:2]
    _solve_lower_kernel[(triton.cdiv(rows, 128),)](matrix, rhs, output,
        rows, depth, 128, enable_fp_fusion=False)
    return output


def solve_cholesky(matrix, rhs):
    rhs = rhs.contiguous()
    output = torch.empty_like(rhs)
    rows, depth = matrix.shape[:2]
    _solve_cholesky_kernel[(triton.cdiv(rows, 128),)](matrix, rhs, output,
        rows, depth, 128, enable_fp_fusion=False)
    return output


def residual_correlation(correlation, coefficients, gram, support):
    output = torch.empty_like(correlation)
    _residual_kernel[(triton.cdiv(correlation.numel(), 256),)](
        correlation, coefficients, gram, support, output, correlation.numel(),
        correlation.shape[-1], coefficients.shape[-1], 256, enable_fp_fusion=False)
    return output


def support_probabilities(residual, available, temperature):
    output = torch.empty_like(residual)
    vocabulary = residual.shape[-1]
    block = triton.next_power_of_2(vocabulary)
    _support_probabilities_kernel[(len(residual),)](residual, available, output,
        vocabulary, temperature, block, num_warps=8 if block>=2048 else 4,
        enable_fp_fusion=False)
    return output


def extend_cholesky(matrix, gram, support, atom):
    rows,previous=support.shape
    output=gram.new_empty(rows,previous+1,previous+1)
    invalid=torch.empty(rows,device=gram.device,dtype=torch.bool)
    _extend_cholesky_kernel[(triton.cdiv(rows,128),)](
        gram if matrix is None else matrix,gram,support,atom,output,invalid,
        rows,previous,gram.shape[-1],128,enable_fp_fusion=False)
    return output,invalid
