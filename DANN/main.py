import pickle
import random
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
from my_data_loader import GetLoader
from model import CNNModel
from test import mytest

batch_size = 512
mu = 1e-1
lr = 5e-2
n_epoch = 100


def train(source, target):
    source_dataset_name = 'source'
    target_dataset_name = 'target'
    model_root = 'models'
    cuda = True
    cudnn.benchmark = True

    manual_seed = random.randint(1, 10000)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    # load model

    my_net = CNNModel()

    # setup optimizer

    optimizer = optim.Adam(my_net.parameters(), lr=lr)

    loss_class = torch.nn.CrossEntropyLoss()
    # torch.nn.AdaptiveLogSoftmaxWithLoss

    loss_domain = torch.nn.CrossEntropyLoss()

    if cuda:
        my_net = my_net.cuda()
        loss_class = loss_class.cuda()
        loss_domain = loss_domain.cuda()

    for p in my_net.parameters():
        p.requires_grad = True

    # training
    best_accu_t = 0.0
    for epoch in range(n_epoch):

        len_dataloader = min(len(source), len(target))
        data_source_iter = iter(source)
        data_target_iter = iter(target)

        for i in range(len_dataloader):

            p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # training model using source data
            data_source = data_source_iter.next()
            s_img, s_label = data_source
            s_label = s_label.long()

            my_net.zero_grad()
            batch_size = len(s_label)

            domain_label = torch.zeros(batch_size).long()

            if cuda:
                s_img = s_img.cuda()
                s_label = s_label.cuda()
                domain_label = domain_label.cuda()

            class_output, domain_output = my_net(input_data=s_img, alpha=alpha)
            err_s_label = loss_class(class_output, s_label)
            err_s_domain = loss_domain(domain_output, domain_label)

            # training model using target data
            data_target = data_target_iter.next()
            t_img, _ = data_target

            batch_size = len(t_img)

            domain_label = torch.ones(batch_size).long()

            if cuda:
                t_img = t_img.cuda()
                domain_label = domain_label.cuda()

            _, domain_output = my_net(input_data=t_img, alpha=alpha)
            err_t_domain = loss_domain(domain_output, domain_label)
            # print(err_t_domain, err_s_domain)
            err = mu * (err_t_domain + err_s_domain) + err_s_label
            err.backward()
            optimizer.step()

            # sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
            #                  % (epoch, i + 1, len_dataloader, err_s_label.data.cpu().numpy(),
            #                     err_s_domain.data.cpu().numpy(), err_t_domain.data.cpu().item()))
            # sys.stdout.flush()
            torch.save(my_net, '{0}/SEED_current.pth'.format(model_root))

        # print('\n')
        accu_s = mytest(source_dataset_name)
        # print('Accuracy of the %s dataset: %f' % ('source', accu_s))
        accu_t = mytest(target_dataset_name)
        # print('Accuracy of the %s dataset: %f\n' % ('target', accu_t))
        if accu_t > best_accu_t:
            best_accu_s = accu_s
            best_accu_t = accu_t
            torch.save(my_net, '{0}/SEED_best.pth'.format(model_root))

    print('============ Summary ============= \n')
    print('Accuracy of the %s dataset: %f' % ('source', best_accu_s))
    print('Accuracy of the %s dataset: %f' % ('target', best_accu_t))
    print('Corresponding model was save in ' + model_root + '/SEED_best.pth')
    accu_test = mytest("target")
    print('============ Test ============= \n')
    print('Accuracy of the %s dataset: %f\n' % ('test', accu_test))
    return accu_test


acc = 0
for i in range(5):
    f = open('dataset/SEED/data.pkl', 'rb')
    data = pickle.load(f)
    f1 = open('dataset/SEED/source.pkl', 'wb')
    f2 = open('dataset/SEED/target.pkl', 'wb')
    source = {}
    target = {}
    target["sub_" + str(i)] = data["sub_" + str(i)]
    data.pop("sub_" + str(i))
    flag = False
    for item in list(data.keys()):
        source[item]=data[item]
    pickle.dump(source, f1)
    pickle.dump(target, f2)
    f1.close()
    f2.close()

    dataset_source = GetLoader(
        data_root=os.path.join('dataset', 'SEED'),
        data_list='source.pkl',
    )

    dataloader_source = torch.utils.data.DataLoader(
        dataset=dataset_source,
        batch_size=batch_size,
        shuffle=True,
    )

    dataset_target = GetLoader(
        data_root=os.path.join('dataset', 'SEED'),
        data_list='target.pkl',
    )

    dataloader_target = torch.utils.data.DataLoader(
        dataset=dataset_target,
        batch_size=batch_size,
        shuffle=True,
    )
    print("第" + str(i) + "折")
    acc += train(dataloader_source, dataloader_target)

print(acc / 5)
