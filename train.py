import multiprocessing
import pickle
import sys
from comet_ml import Experiment
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import BASE_LR, TRAIN_EPOCHS, BATCH_SIZE, DEVICE, MAX_STEPS, USE_SELF_ATTENTION, \
    USE_MEMORY_GATE, MAC_UNIT_DIM, NUM_HEADS
from dataset import CLEVR, collate_data, transform, GQA
from model import MACNetwork
from utils import params_to_dic
from args import parse_args



def train(epoch, dataset_type):
    root = args.root
    if dataset_type == "CLEVR":
        dataset_object = CLEVR(root, transform=transform)
    else:
        dataset_object = GQA(root, transform=transform)

    train_set = DataLoader(
        dataset_object, batch_size=BATCH_SIZE, num_workers=multiprocessing.cpu_count(), collate_fn=collate_data
    )

    dataset = iter(train_set)
    pbar = tqdm(dataset)
    running_loss = 0
    correct_counts = 0
    total_counts = 0

    net.train()
    for image, question, q_len, answer in pbar:
        image, question, answer = (
            image.to(DEVICE),
            question.to(DEVICE),
            answer.to(DEVICE),
        )
        net.zero_grad()
        output = net(image, question, q_len)
        loss = criterion(output, answer)
        loss.backward()
        optimizer.step()

        correct = output.detach().argmax(1) == answer
        correct_counts += sum(correct).item()
        total_counts += image.size(0)

        correct = correct.clone().type(torch.FloatTensor).detach().sum() / BATCH_SIZE
        running_loss += loss.item() / BATCH_SIZE

        pbar.set_description(
            '[Training] Epoch: {}; Loss: {:.8f}; Acc: {:.5f}'.format(epoch + 1, loss.item(), correct))

    print('[Training] loss: {:8f}, accuracy: {:5f}'.format(running_loss / len(train_set.dataset),
                                                         correct_counts / total_counts))
    dataset_object.close()
    return running_loss / len(train_set.dataset), correct_counts / total_counts


def valid(epoch, dataset_type):
    root = args.root
    if dataset_type == "CLEVR":
        dataset_object = CLEVR(root, 'val', transform=None)
    else:
        dataset_object = GQA(root, 'val', transform=None)

    valid_set = DataLoader(dataset_object, batch_size=BATCH_SIZE, num_workers=multiprocessing.cpu_count(),
                           collate_fn=collate_data)
    dataset = iter(valid_set)

    net.eval()
    correct_counts = 0
    total_counts = 0
    running_loss = 0.0
    with torch.no_grad():
        pbar = tqdm(dataset)
        for image, question, q_len, answer in pbar:
            image, question, answer = (
                image.to(DEVICE),
                question.to(DEVICE),
                answer.to(DEVICE),
            )

            output = net(image, question, q_len)
            loss = criterion(output, answer)
            correct = output.detach().argmax(1) == answer
            correct_counts += sum(correct).item()
            total_counts += image.size(0)
            running_loss += loss.item() / BATCH_SIZE

            pbar.set_description(
                '[Val] Epoch: {}; Loss: {:.8f}; Acc: {:.5f}'.format(epoch + 1, loss.item(), correct_counts / total_counts))

    print('[Val] loss: {:8f}, accuracy: {:5f}'.format(running_loss / len(valid_set.dataset),
                                                         correct_counts / total_counts))

    dataset_object.close()
    return running_loss / len(valid_set.dataset), correct_counts / total_counts


if __name__ == '__main__':
    args = parse_args()
    dataset_type = args.model
    root = args.root
    if args.comet:
        experiment = Experiment(api_key='VD0MYyhx0BQcWhxWvLbcalX51',
                        project_name="MAC")
        experiment.set_name(args.exp_name)
        experiment.log_parameters(params_to_dic())

    with open(f'{root}/dic.pkl', 'rb') as f:
        dic = pickle.load(f)

    n_words = len(dic['word_dic']) + 1
    n_answers = len(dic['answer_dic'])

    net = MACNetwork(n_words, MAC_UNIT_DIM[dataset_type], classes=n_answers, max_step=MAX_STEPS,
                     self_attention=USE_SELF_ATTENTION, memory_gate=USE_MEMORY_GATE, num_heads=NUM_HEADS)
    net = nn.DataParallel(net)
    net.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=BASE_LR)
    best_loss = float('inf')
    best_acc = float('inf')
    best_epoch = float('inf')

    for epoch in range(TRAIN_EPOCHS):
        if args.comet:
            with experiment.train(): 
                train_loss, train_acc = train(epoch, dataset_type)
                experiment.log_metrics({'Loss' : train_loss, 'Accuracy': train_acc})
            with experiment.test(): 
                val_loss, val_acc = valid(epoch, dataset_type)
                experiment.log_metrics({'Loss' : val_loss, 'Accuracy': val_acc})
            if val_loss < best_loss:
                with open('checkpoint/{}_best_checkpoint.pth'.format(args.exp_name), 'wb') as f:
                    torch.save(net.state_dict(), f)
                with open('optimizer/{}_best_  optimizer.pth'.format(args.exp_name), 'wb') as f:
                    torch.save(optimizer.state_dict(), f)
                print(f'---- Saving best model weights and optimizer for epoch {epoch + 1} ----')
                best_loss = val_loss
                best_acc = val_acc
                best_epoch = epoch
        else:
            train(epoch, dataset_type)
            valid(epoch, dataset_type)
    print(f'Best epoch {best_epoch} with {best_acc} accuracy and {best_loss} loss')
    print('Training finished')

        
