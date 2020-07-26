def cnf_density(model):
    with torch.no_grad():
        npts = 200
        side = np.linspace(-2., 2., npts)
        xx, yy = np.meshgrid(side, side)
        memory= 100

        x = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])
        x = torch.from_numpy(x).type(torch.float32).to(device)

        z, delta_logp = [], []
        inds = torch.arange(0, x.shape[0]).to(torch.int64)
        for ii in torch.split(inds, int(memory**2)):
            z_full = model(x[ii]).cpu().detach()
            z_, delta_logp_ = z_full[:, 1:], z_full[:, 0]
            z.append(z_)
            delta_logp.append(delta_logp_)

        z = torch.cat(z, 0)
        delta_logp = torch.cat(delta_logp, 0)

        logpz = prior.log_prob(z.cuda()).cpu() # logp(z)
        logpx = logpz - delta_logp
        px = np.exp(logpx.cpu().numpy()).reshape(npts, npts)
        plt.imshow(px, cmap='inferno', vmax=px.mean());
a = cnf_density(model)   


data = ToyDataset()
n_samples = 1 << 16
n_gaussians = 7

X, yn = data.generate(n_samples // n_gaussians, 'gaussians_spiral', n_gaussians=32, n_gaussians_per_loop=10, std_gaussians_start=0.5, std_gaussians_end=0.01, 
                      dim=2, radius_start=20, radius_end=0.1)
X = (X - X.mean())/X.std()

import matplotlib.pyplot as plt
plt.figure(figsize=(3, 3))
plt.scatter(X[:,0], X[:,1], c='orange', alpha=0.3, s=4)