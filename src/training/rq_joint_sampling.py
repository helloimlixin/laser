"""Validate the native RQ sampler's per-depth integer vocabulary limits."""
import numbers


def joint_top_k(value, depth, vocabulary):
    if depth < 1 or vocabulary < 1:
        raise ValueError('Positive residual depth and vocabulary are required')
    if value is None:
        return [vocabulary]
    if isinstance(value,numbers.Integral) and not isinstance(value,bool):
        values=[value]
    else:
        if isinstance(value,(str,bytes)):
            raise ValueError('Top-k must be an integer or one integer per depth')
        try:values=list(value)
        except TypeError as error:raise ValueError('Top-k must be an integer or one integer per depth') from error
    if (len(values) not in (1,depth) or any(isinstance(k,bool) or
            not isinstance(k,numbers.Integral) or k < 1 for k in values)):
        raise ValueError('Use one positive integer or one positive integer per residual depth')
    return [min(int(k),vocabulary) for k in values]
