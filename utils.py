import torch.nn.functional as F
from torch.autograd import Variable


def train(args, epoch, dense_net, loader_train, optimizer, pfile_train):
    dense_net.train()
    nProcessed = 0
    nTrain = len(loader_train.dataset)
    for batch_idx, (data, target) in enumerate(loader_train):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = dense_net(data)
        loss = F.nll_loss(output, target)
        # make_graph.save('/tmp/t.dot', loss.creator); assert(False)
        loss.backward()
        optimizer.step()
        nProcessed += len(data)
        pred = output.data.max(1)[1] # get the index of the max log-probability
        incorrect = pred.ne(target.data).cpu().sum()
        err = 100.*incorrect/len(data)
        partialEpoch = epoch + batch_idx / len(loader_train) - 1
        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tError: {:.6f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(loader_train),
            loss.item(), err))

        pfile_train.write('{},{},{}\n'.format(partialEpoch, loss.item(), err))
        pfile_train.flush()

def test(args, epoch, dense_net, loader_test, pfile_test):
    dense_net.eval()
    test_loss = 0
    incorrect = 0
    for data, target in loader_test:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = dense_net(data)
        test_loss += F.nll_loss(output, target).item()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        incorrect += pred.ne(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(loader_test) # loss function already averages over batch size
    nTotal = len(loader_test.dataset)
    err = 100.*incorrect/nTotal
    print('\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.0f}%)\n'.format(
        test_loss, incorrect, nTotal, err))

    pfile_test.write('{},{},{}\n'.format(epoch, test_loss, err))
    pfile_test.flush()

def adjust_optimizer(alg, alg_param, epoch):
    if alg == 'sgd':
        if epoch < 150: lr = 1e-1
        elif epoch == 150: lr = 1e-2
        elif epoch == 225: lr = 1e-3
        else: return

        for param_group in alg_param.param_groups:
            param_group['lr'] = lr
