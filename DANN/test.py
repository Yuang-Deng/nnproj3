import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms
from my_data_loader import GetLoader
from torchvision import datasets


def mytest(dataset_name):

    model_root = 'models'
    image_root = os.path.join('dataset', 'SEED')

    cuda = True
    cudnn.benchmark = True
    batch_size = 128
    image_size = 28
    alpha = 0

    """load data"""

    if dataset_name == 'source':
        dataset = GetLoader(
            data_root=image_root,
            data_list='source.pkl',
        )
    elif dataset_name == 'test':
        dataset = GetLoader(
            data_root=image_root,
            data_list='test.pkl',
        )
    else:
        dataset = GetLoader(
            data_root=image_root,
            data_list='target.pkl',
        )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    """ test """

    my_net = torch.load(os.path.join(
        model_root, 'SEED_current.pth'
    ))
    my_net = my_net.eval()

    if cuda:
        my_net = my_net.cuda()

    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)

    i = 0
    n_total = 0
    n_correct = 0

    while i < len_dataloader:

        # test model using target data
        data_target = data_target_iter.next()
        t_img, t_label = data_target

        batch_size = len(t_label)

        if cuda:
            t_img = t_img.cuda()
            t_label = t_label.cuda()

        class_output, _ = my_net(input_data=t_img, alpha=alpha)
        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(t_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

        i += 1

    accu = n_correct.data.numpy() * 1.0 / n_total

    return accu
