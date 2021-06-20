# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
General plotting utilities
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from mpl_toolkits.mplot3d import Axes3D


def plot_2d_boundary(model, X, y, mesh, num_classes=2, figsize=(8,4), alpha=0.8):
    """Plots decision boundary of a 2-dimensional task

     :param model: model
     :type model: nn.Module
     :param X: input datasets
     :type X: torch.Tensor
     :param y: input labels
     :type y: torch.Tensor
     :param mesh: meshgrid of points to classify with `model`
     :type mesh: torch.Tensor
     :param num_classes: number of classes
     :type num_classes: int
     :param figsize: figure size
     :type figsize: tuple(int, int)
     :param alpha: alpha of figure
     :type alpha: float
     """
    preds = torch.argmax(nn.Softmax(1)(model(mesh)), dim=1)
    preds = preds.detach().cpu().reshape(mesh.size(0), mesh.size(1))
    plt.figure(figsize=figsize)
    plt.contourf(torch.linspace(0, mesh.size(0), mesh.size(0)), torch.linspace(0, mesh.size(1), mesh.size(1)),
                 preds, cmap='winter', alpha=alpha, levels=10)
    for i in range(num_classes):
        plt.scatter(X[y==i,0], X[y==i,1], alpha=alpha)


def plot_2d_flows(trajectory, num_flows=2, figsize=(8,4), alpha=0.8):
    """Plots datasets flows learned by a neural differential equation.

     :param trajectory: tensor of datasets flows. Assumed to be of dimensions `L, B, *` with `L`:length of trajectory, `B`:batch size, `*`:remaining dimensions.
     :type trajectory: torch.Tensor
     :param num_flows: number of datasets flows to visualize
     :type num_flows: int
     :param figsize: figure size
     :type figsize: tuple(int, int)
     :param alpha: alpha of figure
     :type alpha: float
     """
    plt.figure(figsize=figsize)
    plt.subplot(121)
    plt.title('Dimension: 0')
    for i in range(num_flows):
        plt.plot(trajectory[:,i,0], color='red', alpha=alpha)
    plt.subplot(122)
    plt.title('Dimension: 1')
    for i in range(num_flows):
        plt.plot(trajectory[:,i,1], color='blue', alpha=alpha)


defaults_1D = {'n_grid':100, 'n_levels':30, 'x_span':[-1,1],
            'contour_alpha':0.7, 'cmap':'winter',
            'traj_color':'orange', 'traj_alpha':0.1,
            'device':'cuda:0'}

def plot_traj_vf_1D(model, s_span, traj, device, x_span, n_grid,
                    n_levels=30, contour_alpha=0.7, cmap='winter', traj_color='orange', traj_alpha=0.1):
    """Plots 1D datasets flows.

     :param model: model
     :type model: nn.Module
     :param s_span: number of datasets flows to visualize
     :type s_span: torch.Tensor
     :param traj: figure size
     :type traj: tuple(int, int)
     :param device: alpha of figure
     :type device: float
     :param x_span: alpha of figure
     :type x_span: float
     :param n_grid: alpha of figure
     :type n_grid: float
     """
    ss = torch.linspace(s_span[0], s_span[-1], n_grid)
    xx = torch.linspace(x_span[0], x_span[-1], n_grid)

    S, X = torch.meshgrid(ss,xx)

    if model.controlled:
        ax = st['ax']
        u_traj = traj[0,:,0].repeat(traj.shape[1],1)
        e = torch.abs(st['y'].T - traj[:,:,0])
        color = plt.cm.coolwarm(e)
        for i in range(traj.shape[1]):
            tr = ax.scatter(s_span, u_traj[:,i],traj[:,i,0],
                        c=color[:,i],alpha=1, cmap=color[:,i],zdir='z')
        norm = mpl.colors.Normalize(e.min(),e.max())
        plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='coolwarm'),
             label='Approximation Error', orientation='horizontal')
        ax.set_xlabel(r"$s$ [depth]")
        ax.set_ylabel(r"$u$")
        ax.set_zlabel(r"$h(s)$")
        # make the panes transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # make the grid lines transparent
        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)


    else:
        U, V = torch.ones(n_grid, n_grid), torch.zeros(n_grid, n_grid)
        for i in range(n_grid):
            for j in range(n_grid):
                V[i,j] = model.defunc(
                            S[i,j].reshape(1,-1).to(device),
                            X[i,j].reshape(1,-1).to(device)
                            ).detach().cpu()
        F = torch.sqrt(U**2 + V**2)

        plt.contourf(S,X,F,n_levels,cmap=cmap,alpha=contour_alpha)
        plt.streamplot(S.T.numpy(),X.T.numpy(),
                       U.T.numpy(),V.T.numpy(),
                       color='black',linewidth=1)
        if not traj==None:
            plt.plot(s_span, traj[:,:,0],
                     color=traj_color,alpha=traj_alpha)

        plt.xlabel(r"$s$ [Depth]")
        plt.ylabel(r"$h(s)$")

        return (S, X, U, V)

def plot_2D_depth_trajectory(s_span, trajectory, yn, n_lines):
    color=['orange', 'blue']

    fig = plt.figure(figsize=(8,2))
    ax0 = fig.add_subplot(121)
    ax1 = fig.add_subplot(122)
    for i in range(n_lines):
        ax0.plot(s_span, trajectory[:,i,0], color=color[int(yn[i])], alpha=.1)
        ax1.plot(s_span, trajectory[:,i,1], color=color[int(yn[i])], alpha=.1)

    ax0.set_xlabel(r"$s$ [Depth]")
    ax0.set_ylabel(r"$h_0(s)$")
    ax0.set_title("Dimension 0")
    ax1.set_xlabel(r"$s$ [Depth]")
    ax1.set_ylabel(r"$h_1(s)$")
    ax1.set_title("Dimension 1")


def plot_2D_state_space(trajectory, yn, n_lines):
    color=['orange', 'blue']

    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot(111)
    for i in range(n_lines):
        ax.plot(trajectory[:,i,0], trajectory[:,i,1], color=color[int(yn[i])], alpha=.1);

    ax.set_xlabel(r"$h_0$")
    ax.set_ylabel(r"$h_1$")
    ax.set_title("Flows in the state-space")


def plot_2D_space_depth(s_span, trajectory, yn, n_lines):
    colors = ['orange', 'blue']
    fig = plt.figure(figsize=(6,3))
    ax = Axes3D(fig)
    for i in range(n_lines):
        ax.plot(s_span, trajectory[:,i,0], trajectory[:,i,1], color=colors[yn[i].int()], alpha = .1)
        ax.view_init(30, -110)

    ax.set_xlabel(r"$s$ [Depth]")
    ax.set_ylabel(r"$h_0$")
    ax.set_zlabel(r"$h_1$")
    ax.set_title("Flows in the space-depth")
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)


def plot_static_vector_field(model, trajectory, t=0., N=50, device='cuda'):
    x = torch.linspace(trajectory[:,:,0].min(), trajectory[:,:,0].max(), N)
    y = torch.linspace(trajectory[:,:,1].min(), trajectory[:,:,1].max(), N)
    X, Y = torch.meshgrid(x,y)
    U, V = torch.zeros(N,N), torch.zeros(N,N)

    for i in range(N):
        for j in range(N):
            p = torch.cat([X[i,j].reshape(1,1), Y[i,j].reshape(1,1)],1).to(device)
            O = model.defunc(t,p).detach().cpu()
            U[i,j], V[i,j] = O[0,0], O[0,1]

    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot(111)
    ax.contourf(X, Y, torch.sqrt(U**2 + V**2), cmap='RdYlBu')
    ax.streamplot(X.T.numpy(),Y.T.numpy(),U.T.numpy(),V.T.numpy(), color='k')

    ax.set_xlim([x.min(),x.max()])
    ax.set_ylim([y.min(),y.max()])
    ax.set_xlabel(r"$h_0$")
    ax.set_ylabel(r"$h_1$")
    ax.set_title("Learned Vector Field")


def plot_3D_dataset(X, yn):
    colors = ['orange', 'blue']
    fig = plt.figure(figsize=(4,4))
    ax = Axes3D(fig)
    for i in range(len(X)):
        ax.scatter(X[:,0],X[:,1],X[:,2], color=colors[yn[i].int()], alpha = .1)
    ax.set_xlabel(r"$h_0$")
    ax.set_ylabel(r"$h_1$")
    ax.set_zlabel(r"$h_2$")
    ax.set_title("Data Points")
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
