import torch
from torch._six import container_abcs, istuple
import torch.testing
from itertools import product
import warnings


def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.data.zero_()
    elif isinstance(x, container_abcs.Iterable):
        for elem in x:
            zero_gradients(elem)


def make_jacobian(input, num_out):
    if isinstance(input, torch.Tensor):
        if not input.is_floating_point():
            return None
        if not input.requires_grad:
            return None
        return torch.zeros(input.nelement(), num_out, dtype=input.dtype)
    elif isinstance(input, container_abcs.Iterable) and not isinstance(input, str):
        jacobians = list(filter(
            lambda x: x is not None, (make_jacobian(elem, num_out) for elem in input)))
        if not jacobians:
            return None
        return type(input)(jacobians)
    else:
        return None


def iter_tensors(x, only_requiring_grad=False):
    if isinstance(x, torch.Tensor):
        if x.requires_grad or not only_requiring_grad:
            yield x
    elif isinstance(x, container_abcs.Iterable) and not isinstance(x, str):
        for elem in x:
            for result in iter_tensors(elem, only_requiring_grad):
                yield result


def get_numerical_jacobian(fn, input, target=None, eps=1e-3):
    """
    input: input to `fn`
    target: the Tensors wrt whom Jacobians are calculated (default=`input`)

    Note that `target` may not even be part of `input` to `fn`, so please be
    **very careful** in this to not clone `target`.
    """
    if target is None:
        target = input
    output_size = fn(input).numel()
    jacobian = make_jacobian(target, output_size)

    # It's much easier to iterate over flattened lists of tensors.
    # These are reference to the same objects in jacobian, so any changes
    # will be reflected in it as well.
    x_tensors = [t for t in iter_tensors(target, True)]
    j_tensors = [t for t in iter_tensors(jacobian)]

    # TODO: compare structure
    for x_tensor, d_tensor in zip(x_tensors, j_tensors):
        # need data here to get around the version check because without .data,
        # the following code updates version but doesn't change content
        if x_tensor.is_sparse:
            def get_stride(size):
                dim = len(size)
                tmp = 1
                stride = [0] * dim
                for i in reversed(range(dim)):
                    stride[i] = tmp
                    tmp *= size[i]
                return stride

            x_nnz = x_tensor._nnz()
            x_size = list(x_tensor.size())
            x_indices = x_tensor._indices().t()
            x_values = x_tensor._values().data
            x_stride = get_stride(x_size)

            for i in range(x_nnz):
                x_value = x_values[i]
                for x_idx in product(*[range(m) for m in x_values.size()[1:]]):
                    indices = x_indices[i].tolist() + list(x_idx)
                    d_idx = sum(indices[k] * x_stride[k] for k in range(len(x_size)))
                    orig = x_value[x_idx].item()
                    x_value[x_idx] = orig - eps
                    outa = fn(input).clone()
                    x_value[x_idx] = orig + eps
                    outb = fn(input).clone()
                    x_value[x_idx] = orig
                    r = (outb - outa) / (2 * eps)
                    d_tensor[d_idx] = r.detach().reshape(-1)
        elif x_tensor.layout == torch._mkldnn:
            if len(input) != 1:
                raise ValueError('gradcheck currently only supports functions with 1 input, but got: ',
                                 len(input))
            x_tensor = x_tensor.data
            for d_idx, x_idx in enumerate(product(*[range(m) for m in x_tensor.size()])):
                # this is really inefficient, but without indexing implemented, there's
                # not really a better way than converting back and forth
                x_tensor_dense = x_tensor.to_dense()
                orig = x_tensor_dense[x_idx].item()

                x_tensor_dense[x_idx] = orig - eps
                x_tensor_mkl = x_tensor_dense.to_mkldnn()
                outa = fn([x_tensor_mkl])

                x_tensor_dense[x_idx] = orig + eps
                x_tensor_mkl = x_tensor_dense.to_mkldnn()
                outb = fn([x_tensor_mkl])

                r = (outb - outa) / (2 * eps)
                d_tensor[d_idx] = r.detach().reshape(-1)
        else:
            x_tensor = x_tensor.data
            for d_idx, x_idx in enumerate(product(*[range(m) for m in x_tensor.size()])):
                orig = x_tensor[x_idx].item()
                x_tensor[x_idx] = orig - eps
                outa = fn(input).clone()
                x_tensor[x_idx] = orig + eps
                outb = fn(input).clone()
                x_tensor[x_idx] = orig
                r = (outb - outa) / (2 * eps)
                d_tensor[d_idx] = r.detach().reshape(-1)

    return jacobian


def get_analytical_jacobian(input, output):
    # it is easier to call to_dense() on the sparse output than
    # to modify analytical jacobian
    if output.is_sparse:
        raise ValueError('Sparse output is not supported at gradcheck yet. '
                         'Please call to_dense() on the output of fn for gradcheck.')
    if output.layout == torch._mkldnn:
        raise ValueError('MKLDNN output is not supported at gradcheck yet. '
                         'Please call to_dense() on the output of fn for gradcheck.')
    diff_input_list = list(iter_tensors(input, True))
    jacobian = make_jacobian(input, output.numel())
    jacobian_reentrant = make_jacobian(input, output.numel())
    grad_output = torch.zeros_like(output)
    flat_grad_output = grad_output.view(-1)
    reentrant = True
    correct_grad_sizes = True

    for i in range(flat_grad_output.numel()):
        flat_grad_output.zero_()
        flat_grad_output[i] = 1
        for jacobian_c in (jacobian, jacobian_reentrant):
            grads_input = torch.autograd.grad(output, diff_input_list, grad_output,
                                              retain_graph=True, allow_unused=True)
            for jacobian_x, d_x, x in zip(jacobian_c, grads_input, diff_input_list):
                if d_x is not None and d_x.size() != x.size():
                    correct_grad_sizes = False
                elif jacobian_x.numel() != 0:
                    if d_x is None:
                        jacobian_x[:, i].zero_()
                    else:
                        d_x_dense = d_x.to_dense() if not d_x.layout == torch.strided else d_x
                        assert jacobian_x[:, i].numel() == d_x_dense.numel()
                        jacobian_x[:, i] = d_x_dense.contiguous().view(-1)

    for jacobian_x, jacobian_reentrant_x in zip(jacobian, jacobian_reentrant):
        if jacobian_x.numel() != 0 and (jacobian_x - jacobian_reentrant_x).abs().max() != 0:
            reentrant = False

    return jacobian, reentrant, correct_grad_sizes


def _as_tuple(x):
    if istuple(x):
        return x
    elif isinstance(x, list):
        return tuple(x)
    else:
        return x,


def _differentiable_outputs(x):
    return tuple(o for o in _as_tuple(x) if o.requires_grad)


def gradcheck(func, inputs, eps=1e-6, atol=1e-5, rtol=1e-3, raise_exception=True, check_sparse_nnz=False):
    r"""Check gradients computed via small finite differences against analytical
    gradients w.r.t. tensors in :attr:`inputs` that are of floating point type
    and with ``requires_grad=True``.

    The check between numerical and analytical gradients uses :func:`~torch.allclose`.

    .. note::
        The default values are designed for :attr:`input` of double precision.
        This check will likely fail if :attr:`input` is of less precision, e.g.,
        ``FloatTensor``.

    .. warning::
       If any checked tensor in :attr:`input` has overlapping memory, i.e.,
       different indices pointing to the same memory address (e.g., from
       :func:`torch.expand`), this check will likely fail because the numerical
       gradients computed by point perturbation at such indices will change
       values at all other indices that share the same memory address.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a Tensor or a tuple of Tensors
        inputs (tuple of Tensor or Tensor): inputs to the function
        eps (float, optional): perturbation for finite differences
        atol (float, optional): absolute tolerance
        rtol (float, optional): relative tolerance
        raise_exception (bool, optional): indicating whether to raise an exception if
            the check fails. The exception gives more information about the
            exact nature of the failure. This is helpful when debugging gradchecks.
        check_sparse_nnz (bool, optional): if True, gradcheck allows for SparseTensor input,
            and for any SparseTensor at input, gradcheck will perform check at nnz positions only.

    Returns:
        True if all differences satisfy allclose condition
    """
    def fail_test(msg):
        if raise_exception:
            raise RuntimeError(msg)
        return False

    tupled_inputs = _as_tuple(inputs)
    if any(t.is_sparse for t in tupled_inputs if isinstance(t, torch.Tensor)) and not check_sparse_nnz:
        return fail_test('gradcheck expects all tensor inputs are dense when check_sparse_nnz is set to False.')

    # Make sure that gradients are saved for all inputs
    any_input_requiring_grad = False
    some_input_not_requiring_grad = False
    for inp in tupled_inputs:
        if isinstance(inp, torch.Tensor):
            if inp.requires_grad:
                if inp.dtype != torch.float64:
                    warnings.warn(
                        'At least one of the inputs that requires gradient '
                        'is not of double precision floating point. '
                        'This check will likely fail if all the inputs are '
                        'not of double precision floating point. ')
                any_input_requiring_grad = True
            else:
                some_input_not_requiring_grad = True
            inp.retain_grad()
    if not any_input_requiring_grad:
        raise ValueError(
            'gradcheck expects at least one input tensor to require gradient, '
            'but none of the them have requires_grad=True.')
        if some_input_not_requiring_grad:
            raise ValueError(
                'gradcheck expects if at least one input tensor is required gradient, '
                'then all other inputs should have requires_grad=True.')

    func_out = func(*tupled_inputs)
    output = _differentiable_outputs(func_out)

    if not output:
        for i, o in enumerate(func_out):
            def fn(input):
                return _as_tuple(func(*input))[i]
            numerical = get_numerical_jacobian(fn, tupled_inputs, eps=eps)
            for n in numerical:
                if len(torch.nonzero(n)) > 0:
                    return fail_test('Numerical gradient for function expected to be zero')
        return True

    for i, o in enumerate(output):
        if not o.requires_grad:
            continue

        def fn(input):
            return _as_tuple(func(*input))[i]

        analytical, reentrant, correct_grad_sizes = get_analytical_jacobian(tupled_inputs, o)
        numerical = get_numerical_jacobian(fn, tupled_inputs, eps=eps)

        if not correct_grad_sizes:
            return fail_test('Analytical gradient has incorrect size')

        for j, (a, n) in enumerate(zip(analytical, numerical)):
            if a.numel() != 0 or n.numel() != 0:
                if not torch.allclose(a, n, rtol, atol):
                    return fail_test('Jacobian mismatch for output %d with respect to input %d,\n'
                                     'numerical:%s\nanalytical:%s\n' % (i, j, n, a))

        if not reentrant:
            return fail_test('Backward is not reentrant, i.e., running backward with same '
                             'input and grad_output multiple times gives different values, '
                             'although analytical gradient matches numerical gradient')

    # check if the backward multiplies by grad_output
    output = _differentiable_outputs(func(*tupled_inputs))
    if any([o.requires_grad for o in output]):
        diff_input_list = list(iter_tensors(tupled_inputs, True))
        if not diff_input_list:
            raise RuntimeError("no Tensors requiring grad found in input")
        grads_input = torch.autograd.grad(output, diff_input_list, [torch.zeros_like(o) for o in output],
                                          allow_unused=True)
        for gi, i in zip(grads_input, diff_input_list):
            if gi is None:
                continue
            if isinstance(gi, torch.Tensor) and gi.layout != torch.strided:
                if gi.layout != i.layout:
                    return fail_test('grad is incorrect layout')
                if gi.layout == torch.sparse_coo:
                    if gi.sparse_dim() != i.sparse_dim():
                        return fail_test('grad is sparse tensor, but has incorrect sparse_dim')
                    if gi.dense_dim() != i.dense_dim():
                        return fail_test('grad is sparse tensor, but has incorrect dense_dim')
                gi = gi.to_dense()
                i = i.to_dense()
            if not gi.eq(0).all():
                return fail_test('backward not multiplied by grad_output')
            if gi.type() != i.type():
                return fail_test("grad is incorrect type")
            if gi.size() != i.size():
                return fail_test('grad is incorrect size')

    return True


def gradgradcheck(func, inputs, grad_outputs=None, eps=1e-6, atol=1e-5, rtol=1e-3,
                  gen_non_contig_grad_outputs=False, raise_exception=True):
    r"""Check gradients of gradients computed via small finite differences
    against analytical gradients w.r.t. tensors in :attr:`inputs` and
    :attr:`grad_outputs` that are of floating point type and with
    ``requires_grad=True``.

    This function checks that backpropagating through the gradients computed
    to the given :attr:`grad_outputs` are correct.

    The check between numerical and analytical gradients uses :func:`~torch.allclose`.

    .. note::
        The default values are designed for :attr:`input` and
        :attr:`grad_outputs` of double precision. This check will likely fail if
        they are of less precision, e.g., ``FloatTensor``.

    .. warning::
       If any checked tensor in :attr:`input` and :attr:`grad_outputs` has
       overlapping memory, i.e., different indices pointing to the same memory
       address (e.g., from :func:`torch.expand`), this check will likely fail
       because the numerical gradients computed by point perturbation at such
       indices will change values at all other indices that share the same
       memory address.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a Tensor or a tuple of Tensors
        inputs (tuple of Tensor or Tensor): inputs to the function
        grad_outputs (tuple of Tensor or Tensor, optional): The gradients with
            respect to the function's outputs.
        eps (float, optional): perturbation for finite differences
        atol (float, optional): absolute tolerance
        rtol (float, optional): relative tolerance
        gen_non_contig_grad_outputs (bool, optional): if :attr:`grad_outputs` is
            ``None`` and :attr:`gen_non_contig_grad_outputs` is ``True``, the
            randomly generated gradient outputs are made to be noncontiguous
        raise_exception (bool, optional): indicating whether to raise an exception if
            the check fails. The exception gives more information about the
            exact nature of the failure. This is helpful when debugging gradchecks.

    Returns:
        True if all differences satisfy allclose condition
    """
    tupled_inputs = _as_tuple(inputs)

    if grad_outputs is None:
        # If grad_outputs is not specified, create random Tensors of the same
        # shape, type, and device as the outputs
        def randn_like(x):
            y = torch.testing.randn_like(x if x.is_floating_point() else x.double())
            if gen_non_contig_grad_outputs:
                y = torch.testing.make_non_contiguous(y)
            return y.requires_grad_()
        outputs = _as_tuple(func(*tupled_inputs))
        tupled_grad_outputs = tuple(randn_like(x) for x in outputs)
    else:
        tupled_grad_outputs = _as_tuple(grad_outputs)

    num_outputs = len(tupled_grad_outputs)

    def new_func(*args):
        input_args = args[:-num_outputs]
        grad_outputs = args[-num_outputs:]
        outputs = _differentiable_outputs(func(*input_args))
        input_args = tuple(x for x in input_args if isinstance(x, torch.Tensor) and x.requires_grad)
        grad_inputs = torch.autograd.grad(outputs, input_args, grad_outputs, create_graph=True)
        return grad_inputs

    return gradcheck(new_func, tupled_inputs + tupled_grad_outputs, eps, atol, rtol, raise_exception)
