import torch
import gpytorch

# class UncertaintyCovariance(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x1, x2, var1, var2, lengthscale, dist_func):
#         # if any(ctx.needs_input_grad[:2]):
#         #     raise RuntimeError("LaplacianCovariance cannot compute gradients with " "respect to x1 and x2")
#         # if lengthscale.size(-1) > 1:
#         #     raise ValueError("LaplacianCovariance cannot handle multiple lengthscales")
#         # needs_grad = any(ctx.needs_input_grad)

#         # var_dist = dist_func(var1, -var2)
#         # print("var_dist", var_dist.shape)
#         # mean_dist = dist_func(x1, x2)
#         # print("mean_dist", mean_dist.shape)

#         l2 = lengthscale * lengthscale
#         print("l2", l2)
#         print("var1/l2", (var1/l2))
#         var_dist_div_l = dist_func(var1/l2, var2/l2) # dist(var1, var2) / l2
#         print("var_dist_div_l", var_dist_div_l)
#         print("var1+l2", (var1+l2))
#         var_dist_add_l = dist_func(var1+l2, var2+l2) # l2+ dist(var1, var2)
#         print("var_dist_add_l", var_dist_add_l)
#         mean_dist_div_var_add_l = dist_func(x1, x2)/var_dist_add_l # dist(x1,x2) / l+dist(var1, var2)
#         print("mean_dist_div_var_add_l", mean_dist_div_var_add_l)
#         # temp = mean_dist / (l2+var_dist)
#         # print("temp", temp.shape)
#         temp2 = 1.0+var_dist_div_l
#         print("temp2", temp2)

#         covar_mat = torch.pow(temp2, -0.5)*torch.exp(-0.5*mean_dist_div_var_add_l)
#         print("covar_mat", covar_mat.shape)

#         # if needs_grad:
#         #     d_output_d_input = mean_dist.mul_(covar_mat).div_(lengthscale)
#         #     ctx.save_for_backward(d_output_d_input)

#         print(covar_mat)
#         return covar_mat

#     @staticmethod
#     def backward(ctx, grad_output):
#         d_output_d_input = ctx.saved_tensors[0]
#         lengthscale_grad = grad_output * d_output_d_input
#         return None, None, None, None, lengthscale_grad, None


class UncertainKernel(gpytorch.kernels.Kernel):

    has_lengthscale = True

    def forward(self, x1, x2, **params):

        # preprocessing
        x1_mean = torch.unsqueeze(x1[:,0], 1)
        x1_var = torch.unsqueeze(x1[:,1], 1)

        x2_mean = torch.unsqueeze(x2[:,0], 1)
        x2_var = torch.unsqueeze(x2[:,1], 1)

        if len(x1_mean.shape)>2:
            x1_mean = torch.squeeze(x1_mean)
            x1_var = torch.squeeze(x1_var)

            x2_mean = torch.squeeze(x2_mean)
            x2_var = torch.squeeze(x2_var)

        l2 = self.lengthscale * self.lengthscale


        """
        Numerator Only
        """

        var_dist = l2 + x1_var + x2_var#self.covar_dist(x1_var, -x2_var, square_dist=False)

        # print(var_dist.shape)

        # temp = self.lengthscale + var_dist

        x1_mean_ = x1_mean.div(var_dist)
        x2_mean_ = x2_mean.div(var_dist)

        temp2 = self.covar_dist(x1_mean_, x2_mean_, square_dist=True)
        # temp2 = torch.square(x1_mean_ - x2_mean_)

        dist_mat = temp2.div_(-2).exp_()

        return dist_mat



        """
        Denominator Only
        """

        x1_var_ = x1_var.div(l2)
        x2_var_ = x2_var.div(l2)

        denomDist = self.covar_dist(x1_var_, x2_var_, square_dist=False)

        # denomDist = (x1_var + x2_var)
        # # print(denomDist.shape)

        # denomDist = denomDist.div(l2)
        # # print(denomDist.shape)

        # denomDist = self.covar_dist(denomDist, denomDist, square_dist=False)

        denomDist = denomDist.abs().sqrt()
        # print(denomDist.shape)

        # print(dist_mat.shape)
        


        """
        Combined
        """

        return dist_mat.div(denomDist)



        # if len(l2) == 1:
        #     l2 = l2*torch.ones(len(x1_mean[0]))


        # tmp = 0.0
        # tmp2 = 1.0

        # for i in range(0, len(x1_mean[0,:])):
        #     l2 = self.lengthscale[i] * self.lengthscale[i]
        #     d1 = torch.cdist(x1_mean[:,i].reshape(-1,1), x2_mean[:,i].reshape(-1,1))
        #     d2 = torch.cdist(x1_var[:,i].reshape(-1,1), x2_mean[:,i].reshape(-1,1))
        #     tmp+=d1/(l2+d2)
        #     tmp2*=(1.0+d2/l2)  
        # return torch.pow(tmp2,-0.5)*torch.exp(-0.5*tmp)


        # print("l2", l2)
        # print("var1/l2", (x1_var/l2))
        var_dist_div_l = self.covar_dist(x1_var/l2, -(x2_var/l2)) # dist(var1, var2) / l2
        # print("var_dist_div_l", var_dist_div_l)
        # print("torch method", torch.square(torch.cdist(x1_var/l2, -x2_var/l2)))
        # print("var1+l2", (x1_var+l2))
        var_dist_add_l = self.covar_dist(x1_var+l2, -(x2_var+l2)) # l2+ dist(var1, var2)
        # print("var_dist_add_l", var_dist_add_l)
        mean_dist_div_var_add_l = self.covar_dist(x1_mean, x2_mean)/var_dist_add_l # dist(x1,x2) / l+dist(var1, var2)
        # print("mean_dist_div_var_add_l", mean_dist_div_var_add_l)
        # temp = mean_dist / (l2+var_dist)
        # print("temp", temp.shape)
        temp2 = 1.0+var_dist_div_l
        # print("temp2", temp2)

        covar_mat = torch.pow(temp2, -0.5)*torch.exp(-0.5*mean_dist_div_var_add_l)
        # print("covar_mat", covar_mat)
        return covar_mat
    