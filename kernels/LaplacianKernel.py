import torch
import gpytorch

class LaplacianCovariance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, lengthscale, dist_func):
        if any(ctx.needs_input_grad[:2]):
            raise RuntimeError("LaplacianCovariance cannot compute gradients with " "respect to x1 and x2")
        if lengthscale.size(-1) > 1:
            raise ValueError("LaplacianCovariance cannot handle multiple lengthscales")
        needs_grad = any(ctx.needs_input_grad)
        x1_ = x1.div(lengthscale)
        x2_ = x2.div(lengthscale)
        unitless_dist = dist_func(x1_, x2_)
        # clone because inplace operations will mess with what's saved for backward
        unitless_dist_ = unitless_dist.clone() if needs_grad else unitless_dist
        covar_mat = unitless_dist_.div_(-1.0).exp_()
        if needs_grad:
            d_output_d_input = unitless_dist.mul_(covar_mat).div_(lengthscale)
            ctx.save_for_backward(d_output_d_input)
        return covar_mat

    @staticmethod
    def backward(ctx, grad_output):
        d_output_d_input = ctx.saved_tensors[0]
        lengthscale_grad = grad_output * d_output_d_input
        return None, None, lengthscale_grad, None


def postprocess_laplacian(dist_mat):
    return dist_mat.div_(-1).exp_()


class LaplacianKernel(gpytorch.kernels.Kernel):
    r"""
    Computes a covariance matrix based on the Laplacian kernel
    between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`:
    .. math::
       \begin{equation*}
          k_{\text{Laplacian}}(\mathbf{x_1}, \mathbf{x_2}) = \exp \left( - \left(\mathbf{x_1} - \mathbf{x_2}\right)^\top \Theta^{-1} \left(\mathbf{x_1} - \mathbf{x_2}\right) \right)
       \end{equation*}
    where :math:`\Theta` is a :attr:`lengthscale` parameter.
    See :class:`gpytorch.kernels.Kernel` for descriptions of the lengthscale options.
    .. note::
        This kernel does not have an `outputscale` parameter. To add a scaling parameter,
        decorate this kernel with a :class:`gpytorch.kernels.ScaleKernel`.
    Args:
        :attr:`ard_num_dims` (int, optional):
            Set this if you want a separate lengthscale for each
            input dimension. It should be `d` if :attr:`x1` is a `n x d` matrix. Default: `None`
        :attr:`batch_shape` (torch.Size, optional):
            Set this if you want a separate lengthscale for each
            batch of input data. It should be `b` if :attr:`x1` is a `b x n x d` tensor. Default: `torch.Size([])`.
        :attr:`active_dims` (tuple of ints, optional):
            Set this if you want to compute the covariance of only a few input dimensions. The ints
            corresponds to the indices of the dimensions. Default: `None`.
        :attr:`lengthscale_prior` (Prior, optional):
            Set this if you want to apply a prior to the lengthscale parameter.  Default: `None`.
        :attr:`lengthscale_constraint` (Constraint, optional):
            Set this if you want to apply a constraint to the lengthscale parameter. Default: `Positive`.
        :attr:`eps` (float):
            The minimum value that the lengthscale can take (prevents divide by zero errors). Default: `1e-6`.
    Attributes:
        :attr:`lengthscale` (Tensor):
            The lengthscale parameter. Size/shape of parameter depends on the
            :attr:`ard_num_dims` and :attr:`batch_shape` arguments.
    Example:
        >>> x = torch.randn(10, 5)
        >>> # Non-batch: Simple option
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LaplacianKernel())
        >>> # Non-batch: ARD (different lengthscale for each input dimension)
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LaplacianKernel(ard_num_dims=5))
        >>> covar = covar_module(x)  # Output: LazyTensor of size (10 x 10)
        >>>
        >>> batch_x = torch.randn(2, 10, 5)
        >>> # Batch: Simple option
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LaplacianKernel())
        >>> # Batch: different lengthscale for each batch
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LaplacianKernel(batch_shape=torch.Size([2])))
        >>> covar = covar_module(x)  # Output: LazyTensor of size (2 x 10 x 10)
    """

    has_lengthscale = True

    def forward(self, x1, x2, diag=False, **params):
        if (
            x1.requires_grad
            or x2.requires_grad
            or (self.ard_num_dims is not None and self.ard_num_dims > 1)
            or diag
            or params.get("last_dim_is_batch", False)
        ):
            x1_ = x1.div(self.lengthscale)
            x2_ = x2.div(self.lengthscale)
            return self.covar_dist(
                x1_, x2_, square_dist=True, diag=diag, dist_postprocess_func=postprocess_laplacian, postprocess=True, **params
            )
        return LaplacianCovariance.apply(
            x1,
            x2,
            self.lengthscale,
            lambda x1, x2: self.covar_dist(
                x1, x2, square_dist=False, diag=False, dist_postprocess_func=postprocess_laplacian, postprocess=False, **params
            ),
        )