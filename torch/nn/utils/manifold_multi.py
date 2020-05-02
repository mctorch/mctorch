import torch


def multiprod(A, B):
    # Added just to be parallel to manopt/pymanopt implemenetation
    return torch.matmul(A, B)


def multitransp(A):
    # First check if we have been given just one matrix
    if A.dim() == 2:
        return torch.transpose(A, 1, 0)
    return torch.transpose(A, 2, 1)


def multisym(A):
    # Inspired by MATLAB multisym function by Nicholas Boumal.
    return 0.5 * (A + multitransp(A))


def multiskew(A):
    # Inspired by MATLAB multiskew function by Nicholas Boumal.
    return 0.5 * (A - multitransp(A))
    return 0.5 * (A - multitransp(A))
