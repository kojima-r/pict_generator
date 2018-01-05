
import edward as ed
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import six
import tensorflow as tf
from edward.models import Categorical, Dirichlet, Empirical, InverseGamma, \
    MultivariateNormalDiag, Normal, ParamMixture


class Mixed_gauss_model():
    def __init__(self, num_data, num_cluster, vector_dim, num_mcmc_sample):
        self.K = num_cluster
        self.D = vector_dim
        self.N = num_data
        self.pi = Dirichlet(tf.ones(self.K))
        self.mu = Normal(tf.zeros(self.D),
                         tf.ones(self.D),
                         sample_shape=self.K)
        self.sigmasq = InverseGamma(tf.ones(self.D),
                                    tf.ones(self.D),
                                    sample_shape=self.K)
        self.x = ParamMixture(self.pi, {'loc': self.mu,
                                        'scale_diag': tf.sqrt(self.sigmasq)},
                              MultivariateNormalDiag,
                              sample_shape=self.N)
        self.z = self.x.cat
        self.T = num_mcmc_sample  # number of MCMC samples
        self.qpi = Empirical(tf.Variable(tf.ones([self.T, self.K]) / self.K))
        self.qmu = Empirical(tf.Variable(tf.zeros([self.T, self.K, self.D])))
        self.qsigmasq = Empirical(tf.Variable(tf.ones([self.T, self.K, self.D])))
        self.qz = Empirical(tf.Variable(tf.zeros([self.T, self.N], dtype=tf.int32)))


    def fit(self, x_train):
        self.inference = ed.Gibbs({self.pi: self.qpi,
                                   self.mu: self.qmu,
                                   self.sigmasq: self.qsigmasq,
                                   self.z: self.qz},
                                   data={self.x: x_train})
        self.inference.initialize()

        sess = ed.get_session()

        tf.global_variables_initializer().run()

        t_ph = tf.placeholder(tf.int32, [])
        running_cluster_means = tf.reduce_mean(self.qmu.params[:t_ph], 0)

        for _ in range(self.inference.n_iter):
            info_dict = self.inference.update()
            self.inference.print_progress(info_dict)
            t = info_dict['t']
            if t % self.inference.n_print == 0:
                print("\nInferred cluster means:")
                print(sess.run(running_cluster_means, {t_ph: t - 1}))

    def clustering(self, x_data):
        mu_sample = self.qmu.sample(100)
        sigmasq_sample = self.qsigmasq.sample(100)
        x_post = Normal(loc=tf.ones([self.N, 1, 1, 1]) * mu_sample,
                        scale=tf.ones([self.N, 1, 1, 1]) * tf.sqrt(sigmasq_sample))
        x_broadcasted = tf.tile(tf.reshape(x_data,
                                [self.N, 1, 1, self.D]),
                                [1, 100, self.K, 1])

        log_liks = x_post.log_prob(x_broadcasted)
        log_liks = tf.reduce_sum(log_liks, 3)
        log_liks = tf.reduce_mean(log_liks, 1)

        self.clusters = tf.argmax(log_liks, 1).eval()

    def plot_clusters(self, x_data, C):
        fig = plt.figure()
        ax1 = fig.add_subplot(1,3,1)
        ax2 = fig.add_subplot(1,3,2)
        ax3 = fig.add_subplot(1,3,3)

        ax1.scatter(x_data[:, 0], x_data[:, 1])
        ax1.axis([-4, 4, -4, 4])
        #ax1.title("Simulated dataset")

        ax2.scatter(x_data[:, 0], x_data[:, 1], c = C, cmap=cm.Set1)
        ax2.axis([-4, 4, -4, 4])

        ax3.scatter(x_data[:, 0], x_data[:, 1], c=self.clusters, cmap=cm.Set1)
        ax3.axis([-4, 4, -4, 4])
        #ax2.title("Predicted cluster assignments")
        plt.show()




if __name__ == '__main__':
    plt.style.use('ggplot')

    def build_toy_dataset(N):
        pi = np.array([0.2, 0.3, 0.15, 0.2, 0.15])
        mus = [[1, 1], [-1, -1], [-1, 0.7], [1, -0.8], [0, 0]]
        stds = [[0.2, 0.2], [0.2, 0.2], [0.3, 0.1], [0.1, 0.2], [0.05, 0.05]]
        x = np.zeros((N, 2), dtype=np.float32)
        C = []
        for n in range(N):
            k = np.argmax(np.random.multinomial(1, pi))
            x[n, :] = np.random.multivariate_normal(mus[k], np.diag(stds[k]))
            C.append(k)
        return x, np.array(C)

    N = 500  # number of data points
    K = 10  # number of components
    D = 2  # dimensionality of data
    T = 1000 # number of mcmc samples
    ed.set_seed(42)

    x_train, C = build_toy_dataset(N)
    model = Mixed_gauss_model(N,K,D,T)
    model.fit(x_train)
    model.clustering(x_train)
    model.plot_clusters(x_train, C)


