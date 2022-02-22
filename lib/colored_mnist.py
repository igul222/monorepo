import lib.datasets
import torch
  
def _make_environment(images, labels, e):
    images, labels = images.cpu(), labels.cpu()
    def torch_bernoulli(p, size):
        return (torch.rand(size) < p).float()
    def torch_xor(a, b):
        return (a-b).abs()
    images = images.reshape((-1, 28, 28))[:, ::2, ::2]
    labels = (labels < 5).float()
    labels = torch_xor(labels, torch_bernoulli(0.25, len(labels)))
    colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
    images = torch.stack([images, images], dim=1)
    images[torch.tensor(range(len(images))), (1-colors).long(), :, :] *= 0
    return {
        'images': images.view(-1, 392).cuda(),
        'labels': labels[:, None].cuda()
    }

def colored_mnist():
    """Colored MNIST environments, as in the IRM paper."""
    mnist_train = lib.datasets.mnist('train')
    mnist_test = lib.datasets.mnist('test')
    envs = [
        _make_environment(mnist_train[0][::2],  mnist_train[1][::2],  0.2),
        _make_environment(mnist_train[0][1::2], mnist_train[1][1::2], 0.1),
        _make_environment(mnist_test[0],        mnist_test[1],        0.9)
    ]
    return envs
