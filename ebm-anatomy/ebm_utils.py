# download Oxford Flowers 102, plotting functions, and toy dataset

import torch as t
import torchvision as tv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os


##########################
# ## DOWNLOAD FLOWERS ## #
##########################
# Code with minor modification from https://github.com/microsoft/CNTK/tree/master/Examples/Image/DataSets/Flowers
# Original Version: Copyright (c) Microsoft Corporation

def download_flowers_data():
    import tarfile
    try:
        from urllib.request import urlretrieve
    except ImportError:
        from urllib import urlretrieve

    dataset_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/flowers/')
    if not os.path.exists(os.path.join(dataset_folder, "jpg")):
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder)
        print('Downloading data from http://www.robots.ox.ac.uk/~vgg/data/flowers/102/ ...')
        tar_filename = os.path.join(dataset_folder, "102flowers.tgz")
        if not os.path.exists(tar_filename):
            urlretrieve("http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz", tar_filename)

        # extract flower images from tar file
        print('Extracting ' + tar_filename + '...')
        tarfile.open(tar_filename).extractall(path=dataset_folder)

        # clean up
        os.remove(tar_filename)
        print('Done.')
    else:
        print('Data available at ' + dataset_folder)


##################
# ## PLOTTING ## #
##################

# visualize negative samples synthesized from energy
def plot_ims(p, x): tv.utils.save_image(t.clamp(x, -1., 1.), p, normalize=True, nrow=int(x.shape[0] ** 0.5))

# plot diagnostics for learning
def plot_diagnostics(batch, en_diffs, grad_mags, fid_record, f_x_q_record, f_x_st_record, instability_record, exp_dir, fontsize=10):
    # axis tick size
    matplotlib.rc('xtick', labelsize=8)
    matplotlib.rc('ytick', labelsize=8)
    fig = plt.figure(figsize=(14, 8))

    # energy difference
    ax = fig.add_subplot(231)
    ax.plot(en_diffs[0:(batch+1)].data.cpu().numpy())
    ax.axhline(y=0, ls='--', c='k')
    ax.set_title('Energy Difference', fontsize=fontsize)
    ax.set_xlabel('batch', fontsize=fontsize)
    ax.set_ylabel('$d_{s_t}$', fontsize=fontsize)
    
    # mean langevin gradient
    ax = fig.add_subplot(232)
    ax.plot(grad_mags[0:(batch+1)].data.cpu().numpy())
    ax.set_title('Average Langevin Gradient Magnitude', fontsize=fontsize)
    ax.set_xlabel('batch', fontsize=fontsize)
    ax.set_ylabel('$r_{s_t}$', fontsize=fontsize)
    
    # FID score
    ax = fig.add_subplot(233)
    fid_data = fid_record[0:(batch+1)].data.cpu().numpy()
    fid_inds = np.where(np.isfinite(fid_data))[0]
    if len(fid_inds) > 0:
        ax.scatter(fid_inds + 1, fid_data[fid_inds], marker='x', s=35)
    ax.set_title('FID Score', fontsize=fontsize)
    ax.set_xlabel('batch', fontsize=fontsize)
    ax.set_ylabel('FID', fontsize=fontsize)
    
    # Shared y-axis limits for both energy plots
    f_x_q_data = f_x_q_record[0:(batch+1)].data.cpu().numpy()
    f_x_st_data = f_x_st_record[0:(batch+1)].data.cpu().numpy()
    y_min = min(f_x_q_data.min(), f_x_st_data.min())
    y_max = max(f_x_q_data.max(), f_x_st_data.max())
    f_x_q_last = float(f_x_q_data[-1]) if len(f_x_q_data) > 0 else float('nan')
    f_x_st_last = float(f_x_st_data[-1]) if len(f_x_st_data) > 0 else float('nan')

    # Distribution energy
    ax = fig.add_subplot(234)
    ax.plot(f_x_q_data)
    ax.set_ylim(y_min, y_max)
    ax.axhline(y=f_x_q_last, ls='--', c='r', label=f'Last value: {f_x_q_last:.4f}')
    ax.set_title('Distribution Energy', fontsize=fontsize)
    ax.set_xlabel('batch', fontsize=fontsize)
    ax.set_ylabel('Energy', fontsize=fontsize)
    ax.legend(fontsize=max(6, fontsize-1), loc='best')
    
    # Generated energy
    ax = fig.add_subplot(235)
    ax.plot(f_x_st_data)
    ax.set_ylim(y_min, y_max)
    ax.axhline(y=f_x_st_last, ls='--', c='r', label=f'Last value: {f_x_st_last:.4f}')
    ax.set_title('Generated Energy', fontsize=fontsize)
    ax.set_xlabel('batch', fontsize=fontsize)
    ax.set_ylabel('Energy', fontsize=fontsize)
    ax.legend(fontsize=max(6, fontsize-1), loc='best')

    # Instability
    ax = fig.add_subplot(236)
    ax.plot(instability_record[0:(batch+1)].data.cpu().numpy())
    ax.set_title('Instability', fontsize=fontsize)
    ax.set_xlabel('batch', fontsize=fontsize)
    ax.set_ylabel('Instability', fontsize=fontsize)
    
    # save figure
    plt.subplots_adjust(hspace=0.45, wspace=0.3)
    plt.savefig(os.path.join(exp_dir, 'diagnosis_plot.pdf'), format='pdf')
    plt.close()


#####################
# ## TOY DATASET ## #
#####################

class ToyDataset:
    def __init__(self, toy_type='gmm', toy_groups=8, toy_sd=0.15, toy_radius=1, viz_res=500, kde_bw=0.05):
        # import helper functions
        from scipy.stats import gaussian_kde
        from scipy.stats import multivariate_normal
        self.gaussian_kde = gaussian_kde
        self.mvn = multivariate_normal

        # toy dataset parameters
        self.toy_type = toy_type
        self.toy_groups = toy_groups
        self.toy_sd = toy_sd
        self.toy_radius = toy_radius
        self.weights = np.ones(toy_groups) / toy_groups
        if toy_type == 'gmm':
            means_x = np.cos(2*np.pi*np.linspace(0, (toy_groups-1)/toy_groups, toy_groups)).reshape(toy_groups, 1, 1, 1)
            means_y = np.sin(2*np.pi*np.linspace(0, (toy_groups-1)/toy_groups, toy_groups)).reshape(toy_groups, 1, 1, 1)
            self.means = toy_radius * np.concatenate((means_x, means_y), axis=1)
        else:
            self.means = None

        # ground truth density
        if self.toy_type == 'gmm':
            def true_density(x):
                density = 0
                for k in range(toy_groups):
                    density += self.weights[k]*self.mvn.pdf(np.array([x[1], x[0]]), mean=self.means[k].squeeze(),
                                                            cov=(self.toy_sd**2)*np.eye(2))
                return density
        elif self.toy_type == 'rings':
            def true_density(x):
                radius = np.sqrt((x[1] ** 2) + (x[0] ** 2))
                density = 0
                for k in range(toy_groups):
                    density += self.weights[k] * self.mvn.pdf(radius, mean=self.toy_radius * (k + 1),
                                                              cov=(self.toy_sd**2))/(2*np.pi*self.toy_radius*(k+1))
                return density
        else:
            raise RuntimeError('Invalid option for toy_type (use "gmm" or "rings")')
        self.true_density = true_density

        # viz parameters
        self.viz_res = viz_res
        self.kde_bw = kde_bw
        if toy_type == 'rings':
            self.plot_val_max = toy_groups * toy_radius + 4 * toy_sd
        else:
            self.plot_val_max = toy_radius + 4 * toy_sd

        # save values for plotting groundtruth landscape
        self.xy_plot = np.linspace(-self.plot_val_max, self.plot_val_max, self.viz_res)
        self.z_true_density = np.zeros(self.viz_res**2).reshape(self.viz_res, self.viz_res)
        for x_ind in range(len(self.xy_plot)):
            for y_ind in range(len(self.xy_plot)):
                self.z_true_density[x_ind, y_ind] = self.true_density([self.xy_plot[x_ind], self.xy_plot[y_ind]])

    def sample_toy_data(self, num_samples):
        toy_sample = np.zeros(0).reshape(0, 2, 1, 1)
        sample_group_sz = np.random.multinomial(num_samples, self.weights)
        if self.toy_type == 'gmm':
            for i in range(self.toy_groups):
                sample_group = self.means[i] + self.toy_sd * np.random.randn(2*sample_group_sz[i]).reshape(-1, 2, 1, 1)
                toy_sample = np.concatenate((toy_sample, sample_group), axis=0)
        elif self.toy_type == 'rings':
            for i in range(self.toy_groups):
                sample_radii = self.toy_radius*(i+1) + self.toy_sd * np.random.randn(sample_group_sz[i])
                sample_thetas = 2 * np.pi * np.random.random(sample_group_sz[i])
                sample_x = sample_radii.reshape(-1, 1) * np.cos(sample_thetas).reshape(-1, 1)
                sample_y = sample_radii.reshape(-1, 1) * np.sin(sample_thetas).reshape(-1, 1)
                sample_group = np.concatenate((sample_x, sample_y), axis=1)
                toy_sample = np.concatenate((toy_sample, sample_group.reshape(-1, 2, 1, 1)), axis=0)
        else:
            raise RuntimeError('Invalid option for toy_type ("gmm" or "rings")')

        return toy_sample

    def plot_toy_density(self, plot_truth=False, f=None, epsilon=0.0, x_s_t=None, save_path='toy.pdf'):
        num_plots = 0
        if plot_truth:
            num_plots += 1

        # density of learned EBM
        if f is not None:
            num_plots += 1
            xy_plot_torch = t.Tensor(self.xy_plot).view(-1, 1, 1, 1).to(next(f.parameters()).device)
            # y values for learned energy landscape of descriptor network
            z_learned_energy = np.zeros([self.viz_res, self.viz_res])
            for i in range(len(self.xy_plot)):
                y_vals = float(self.xy_plot[i]) * t.ones_like(xy_plot_torch)
                vals = t.cat((xy_plot_torch, y_vals), 1)
                z_learned_energy[i] = f(vals).data.cpu().numpy()
            # rescale y values to correspond to the groundtruth temperature
            if epsilon > 0:
                z_learned_energy *= 2 / (epsilon ** 2)

            # transform learned energy into learned density
            z_learned_density_unnormalized = np.exp(- (z_learned_energy - np.min(z_learned_energy)))
            bin_area = (self.xy_plot[1] - self.xy_plot[0]) ** 2
            z_learned_density = z_learned_density_unnormalized / (bin_area * np.sum(z_learned_density_unnormalized))

        # kernel density estimate of shortrun samples
        if x_s_t is not None:
            num_plots += 1
            density_estimate = self.gaussian_kde(x_s_t.squeeze().cpu().numpy().transpose(), bw_method=self.kde_bw)
            z_kde_density = np.zeros([self.viz_res, self.viz_res])
            for i in range(len(self.xy_plot)):
                for j in range(len(self.xy_plot)):
                    z_kde_density[i, j] = density_estimate((self.xy_plot[j], self.xy_plot[i]))

        # plot results
        plot_ind = 0
        fig = plt.figure()

        # true density
        if plot_truth:
            plot_ind += 1
            ax = fig.add_subplot(2, num_plots, plot_ind)
            ax.set_title('True density')
            plt.imshow(self.z_true_density, cmap='viridis')
            plt.axis('off')
            ax = fig.add_subplot(2, num_plots, plot_ind + num_plots)
            ax.set_title('True log-density')
            plt.imshow(np.log(self.z_true_density + 1e-10), cmap='viridis')
            plt.axis('off')
        # learned ebm
        if f is not None:
            plot_ind += 1
            ax = fig.add_subplot(2, num_plots, plot_ind)
            ax.set_title('EBM density')
            plt.imshow(z_learned_density, cmap='viridis')
            plt.axis('off')
            ax = fig.add_subplot(2, num_plots, plot_ind + num_plots)
            ax.set_title('EBM log-density')
            plt.imshow(np.log(z_learned_density + 1e-10), cmap='viridis')
            plt.axis('off')
        # shortrun kde
        if x_s_t is not None:
            plot_ind += 1
            ax = fig.add_subplot(2, num_plots, plot_ind)
            ax.set_title('Short-run KDE')
            plt.imshow(z_kde_density, cmap='viridis')
            plt.axis('off')
            ax = fig.add_subplot(2, num_plots, plot_ind + num_plots)
            ax.set_title('Short-run log-KDE')
            plt.imshow(np.log(z_kde_density + 1e-10), cmap='viridis')
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
        plt.close()
