import numpy as np
import torch
from art.utils import load_mnist, load_cifar10

def load_model(dataset_name, checkpoints_dir, device):
    if dataset_name == 'mnist':
        from models.small_cnn import SmallCNN
        path = '{}small_cnn.pt'.format(checkpoints_dir)
        model = SmallCNN()
    elif dataset_name == 'cifar10':
        from models.resnet import ResNet18
        path = '{}rn-best.pt'.format(checkpoints_dir)
        model = ResNet18(num_classes=10)

    if torch.cuda.is_available():
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)
        model = model.to(device)
    else:
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)

    return model

def load_data(dataset_name):
    if dataset_name == 'mnist':
        (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

    elif dataset_name == 'cifar10':
        (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()

    x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
    x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)

    return x_train, y_train, x_test, y_test

def get_prediction_by_bs(model, X, num_classes, batch_size=500):
    import math
    preds = np.zeros((X.shape[0], num_classes))
    for i in range(math.ceil(X.shape[0] / batch_size)):
        preds[i * batch_size:(i + 1) * batch_size] = model(
            torch.tensor(X[i * batch_size:(i + 1) * batch_size]).float().cuda()).cpu().detach().numpy()
    return preds