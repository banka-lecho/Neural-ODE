import math
import numpy as np
from IPython.display import clear_output
from tqdm import tqdm_notebook as tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.color_palette("bright")
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torch import Tensor
from torch import nn
from torch.nn  import functional as F 
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()


def ode_solve(z0, t0, t1, f):
    """
    Простейший метод эволюции ОДУ - метод Эйлера
    """
    h_max = 0.05
    n_steps = math.ceil((abs(t1 - t0)/h_max).max().item())

    h = (t1 - t0)/n_steps
    t = t0
    z = z0

    for i_step in range(n_steps):
        z = z + h * f(z, t)
        t = t + h
    return z

class ODEF(nn.Module):
    class NeuralODE(nn.Module):
        def __init__(self, func):
            super(NeuralODE, self).__init__()
            assert isinstance(func, ODEF)
            self.func = func

        def forward(self, z0, t=Tensor([0., 1.]), return_whole_sequence=False):
            t = t.to(z0)
            z = ODEAdjoint.apply(z0, t, self.func.flatten_parameters(), self.func)
            if return_whole_sequence:
                return z
            else:
                return z[-1]

    class ODEAdjoint(torch.autograd.Function):
        @staticmethod
        def forward(ctx, z0, t, flat_parameters, func):
            assert isinstance(func, ODEF)
            bs, *z_shape = z0.size()
            time_len = t.size(0)

            with torch.no_grad():
                z = torch.zeros(time_len, bs, *z_shape).to(z0)
                z[0] = z0
                for i_t in range(time_len - 1):
                    z0 = ode_solve(z0, t[i_t], t[i_t + 1], func)
                    z[i_t + 1] = z0

            ctx.func = func
            ctx.save_for_backward(t, z.clone(), flat_parameters)
            return z

        @staticmethod
        def backward(ctx, dLdz):
            """
            dLdz shape: time_len, batch_size, *z_shape
            """
            func = ctx.func
            t, z, flat_parameters = ctx.saved_tensors
            time_len, bs, *z_shape = z.size()
            n_dim = np.prod(z_shape)
            n_params = flat_parameters.size(0)

            # Динамика аугментированной системы,
            # которую надо эволюционировать обратно во времени
            def augmented_dynamics(aug_z_i, t_i):
                """
                Тензоры здесь - это срезы по времени
                t_i - тензор с размерами: bs, 1
                aug_z_i - тензор с размерами: bs, n_dim*2 + n_params + 1
                """
                # игнорируем параметры и время
                z_i, a = aug_z_i[:, :n_dim], aug_z_i[:, n_dim:2 * n_dim]
                # Unflatten z and a
                z_i = z_i.view(bs, *z_shape)
                a = a.view(bs, *z_shape)
                with torch.set_grad_enabled(True):
                    t_i = t_i.detach().requires_grad_(True)
                    z_i = z_i.detach().requires_grad_(True)

                    faug = func.forward_with_grad(z_i, t_i, grad_outputs=a)
                    func_eval, adfdz, adfdt, adfdp = faug

                    adfdz = adfdz if adfdz is not None else torch.zeros(bs, *z_shape)
                    adfdp = adfdp if adfdp is not None else torch.zeros(bs, n_params)
                    adfdt = adfdt if adfdt is not None else torch.zeros(bs, 1)
                    adfdz = adfdz.to(z_i)
                    adfdp = adfdp.to(z_i)
                    adfdt = adfdt.to(z_i)

                    # Flatten f and adfdz
                    func_eval = func_eval.view(bs, n_dim)
                    adfdz = adfdz.view(bs, n_dim)
                    return torch.cat((func_eval, -adfdz, -adfdp, -adfdt), dim=1)

                dLdz = dLdz.view(time_len, bs, n_dim)  # flatten dLdz для удобства
                with torch.no_grad():
                    ## Создадим плейсхолдеры для возвращаемых градиентов
                    # Распространенные назад сопряженные состояния,
                    # которые надо поправить градиентами от наблюдений
                    adj_z = torch.zeros(bs, n_dim).to(dLdz)
                    adj_p = torch.zeros(bs, n_params).to(dLdz)
                    # В отличие от z и p, нужно вернуть градиенты для всех моментов времени
                    adj_t = torch.zeros(time_len, bs, 1).to(dLdz)

                    for i_t in range(time_len - 1, 0, -1):
                        z_i = z[i_t]
                        t_i = t[i_t]
                        f_i = func(z_i, t_i).view(bs, n_dim)

                        # Рассчитаем прямые градиенты от наблюдений
                        dLdz_i = dLdz[i_t]
                        dLdt_i = torch.bmm(torch.transpose(dLdz_i.unsqueeze(-1), 1, 2),
                                           f_i.unsqueeze(-1))[:, 0]

                        # Подправим ими сопряженные состояния
                        adj_z += dLdz_i
                        adj_t[i_t] = adj_t[i_t] - dLdt_i
                        # Упакуем аугментированные переменные в вектор
                        aug_z = torch.cat((
                            z_i.view(bs, n_dim),
                            adj_z, torch.zeros(bs, n_params).to(z)
                            adj_t[i_t]),
                            dim=-1
                        )

                        # Решим (эволюционируем) аугментированную систему назад во времени
                        aug_ans = ode_solve(aug_z, t_i, t[i_t - 1], augmented_dynamics)

                        # Распакуем переменные обратно из решенной системы
                        adj_z[:] = aug_ans[:, n_dim:2 * n_dim]
                        adj_p[:] += aug_ans[:, 2 * n_dim:2 * n_dim + n_params]
                        adj_t[i_t - 1] = aug_ans[:, 2 * n_dim + n_params:]

                        del aug_z, aug_ans

                        ## Подправим сопряженное состояние в нулевой момент прямыми градиентами
                        # Вычислим прямые градиенты
                    dLdz_0 = dLdz[0]
                    dLdt_0 = torch.bmm(torch.transpose(dLdz_0.unsqueeze(-1), 1, 2),
                                       f_i.unsqueeze(-1))[:, 0]

                    # Подправим
                    adj_z += dLdz_0
                    adj_t[0] = adj_t[0] - dLdt_0
                return adj_z.view(bs, *z_shape), adj_t, adj_p, None