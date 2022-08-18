import argparse
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import Utils
from collections import OrderedDict
import os
from preprocess.Dataset import get_dataloader
# from transformer.Models import Transformer
# import transformer.Constants as Constants
from transformerhkhd.Models import Transformer 
import transformerhkhd.Constants as Constants
from tqdm import tqdm
import random
from datetime import datetime
from pathlib import Path

def find_max_length(lst):
    maxLength = max(len(x) for x in lst)
    return maxLength


def find_min_length(lst):
    minLength = min(len(x) for x in lst)
    return minLength


def prepare_dataloader(opt):
    """ Load data and prepare dataloader. """

    def load_data(name, dict_name):
        with open(name, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')
            num_types = data['dim_process']
            data = data[dict_name]
            return data, int(num_types)

    print('[Info] Loading train data...')
    train_data, num_types = load_data(opt.data + 'train.pkl', 'train')

    max_len_train = find_max_length(train_data)
    min_len_train = find_min_length(train_data)
    print('Max train data {}'.format(max_len_train))
    print('Min train data {}'.format(min_len_train))

    print('[Info] Loading dev data...')
    dev_data, _ = load_data(opt.data + 'dev.pkl', 'dev')

    max_len_dev = find_max_length(dev_data)
    min_len_dev = find_min_length(dev_data)
    print('Max dev data {}'.format(max_len_dev))
    print('Min dev data {}'.format(min_len_dev))

    print('[Info] Loading test data...')
    test_data, _ = load_data(opt.data + 'test.pkl', 'test')
    max_len_test = find_max_length(test_data)
    min_len_test = find_min_length(test_data)
    print('Max test data {}'.format(max_len_test))
    print('Min test data {}'.format(min_len_test))

    trainloader = get_dataloader(train_data, opt.batch_size, opt.num_workers, shuffle=True)
    validationloader = get_dataloader(dev_data, opt.batch_size, opt.num_workers, shuffle=False)
    testloader = get_dataloader(test_data, opt.batch_size, opt.num_workers, shuffle=False)

    # return trainloader, validationloader, testloader, num_types, maxLength
    return trainloader, validationloader, testloader, num_types


def train_epoch(model, training_data, optimizer, pred_loss_func, opt, epoch=None):
    """ Epoch operation in training phase. """

    model.train()


    # check weights
    params = []
    for param in model.parameters():
        params.append(param.view(-1))
    params = torch.cat(params)
    # print(params)
    #
    total_event_ll = 0  # cumulative event log-likelihood
    total_time_se = 0  # cumulative time prediction squared-error
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions
    total_losses = 0

    # For the event and non-event plot
    total_event_loss = 0
    total_non_event_loss = 0

    # Decay pred loss
    pred_loss_prev = 0
    counter = 0


    for batch in tqdm(training_data, mininterval=2,
                      desc='  - (Training)   ', leave=False):
        """ prepare data """
        event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)
        """ forward """
        optimizer.zero_grad()
        torch.autograd.set_detect_anomaly(False)
        enc_out, prediction = model(event_type, event_time)

        """ backward """
        # negative log-likelihood
        event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, event_time, event_type, opt)

        event_loss = -torch.sum(event_ll - non_event_ll)

        # type prediction
        pred_loss, pred_num_event = Utils.type_loss(prediction[0], event_type, pred_loss_func)
        # breakpoint()
        # time prediction
        se = Utils.time_loss(prediction[1], event_time)

        num_event = event_type.ne(Constants.PAD).sum().item()
        num_pred = event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]

        pred_loss = pred_loss / num_pred
        event_loss = event_loss / num_event
        se = se / num_event

        # --------rescale event loss and pred loss to the same magnitude ------

        loss = pred_loss + event_loss
        # scale_time_loss = True
        # scale_time_loss = True
        scale_time_loss = True
        if scale_time_loss:
            ## =========== old scaling ===========
            se_ = se
            loss_scale = np.log10(max(loss.item(), 1e-8))
            se_scale = np.log10(max(se_.item(), 1e-8))
            scale_value = 10 ** np.ceil(se_scale-loss_scale)
            se_ = se_ / scale_value
            loss = loss + se_
        else:
            loss = loss + se

        loss_ = loss / num_pred
        loss_.backward()

        """ update parameters """
        optimizer.step()

        """ note keeping """
        total_event_ll += -event_loss.item()
        total_time_se += se.item()
        total_event_rate += pred_num_event.item()
        total_num_event += event_type.ne(Constants.PAD).sum().item()
        # we do not predict the first event
        total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]

        # This is event ll not event loss
        total_event_loss += event_ll.sum().item()
        total_non_event_loss += non_event_ll.sum().item()
        total_losses += loss.item()


    rmse = np.sqrt(total_time_se / total_num_pred)

    return total_event_ll / total_num_event\
        , total_event_rate / total_num_pred\
        , rmse\
        , total_event_loss / total_num_event, total_non_event_loss / total_num_event \
        , total_losses / total_num_event



def eval_epoch(model, validation_data, pred_loss_func, opt, epoch=None):
    """ Epoch operation in evaluation phase. """

    model.eval()

    total_event_ll = 0  # cumulative event log-likelihood
    total_time_se = 0  # cumulative time prediction squared-error
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions
    total_losses = 0

    total_time_error = []

    # For the event and non-event plot
    total_event_loss = 0
    total_non_event_loss = 0

    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,
                          desc='  - (Validation) ', leave=False):
            """ prepare data """
            event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)

            """ forward """
            enc_out, prediction = model(event_type, event_time, epoch)

            """ compute loss """
            event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, event_time, event_type, opt)
            event_loss = -torch.sum(event_ll - non_event_ll)
            pred_loss, pred_num = Utils.type_loss(prediction[0], event_type, pred_loss_func)
            se = Utils.time_loss(prediction[1], event_time)

            # ----------- scale the the time loss for better balance -------
            loss = pred_loss + event_loss
            # loss_scale = np.ceil(np.log10(max(loss.item(), 1e-8)))
            # se_scale = np.ceil(np.log10(max(se.item(), 1e-8)))
            # scale_value = 10 ** (se_scale-loss_scale)
            scale_time_loss = True
            # scale_time_loss = False
            if scale_time_loss:
                loss_scale = np.log10(max(loss.item(), 1e-8))
                se_scale = np.log10(max(se.item(), 1e-8))
                scale_value = 10 ** np.ceil(se_scale-loss_scale)
                se_ = se / scale_value
            loss = loss + se_

            """ note keeping """
            total_event_ll += -event_loss.item()
            total_time_se += se.item()
            total_event_rate += pred_num.item()
            total_num_event += event_type.ne(Constants.PAD).sum().item()
            total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]
            total_time_error.append(se.item())
            # This is event ll not event loss
            total_event_loss += event_ll.sum().item()
            total_non_event_loss += non_event_ll.sum().item()
            total_losses += loss.item()

    if opt.small_test_fin_num_pred == 2:
        total_time_error = np.asarray(total_time_error)
        rmse = np.sqrt(np.mean(total_time_error), dtype=np.float64)
    else:
        rmse = np.sqrt(total_time_se / total_num_pred)

    return total_event_ll / total_num_event, total_event_rate / total_num_pred, rmse\
        , total_event_loss / total_num_event, total_non_event_loss / total_num_event, total_losses / total_num_event


def train(model, training_data, validation_data, test_data, optimizer, scheduler, pred_loss_func, opt, log_name):
    """ Start training. """
    # early_stop_threshold = 1e-3  # default
    early_stop_threshold = 0
    # patience=3 # default
    # patience= 10
    early_step = 0
    train_event_lls = []
    valid_event_losses = []  # validation log-likelihood
    valid_pred_losses = []  # validation event type prediction accuracy
    valid_rmse = []  # validation event time prediction RMSE
    test_event_losses = []
    test_pred_losses = []
    test_rmse = []

    train_event_loss_total = []
    train_non_event_loss_total = []
    test_event_loss_total = []
    test_non_event_loss_total = []
    valid_event_loss_total = []
    valid_non_event_loss_total = []

    valid_total_losses = []

    best_state_dict = None

    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')

        start = time.time()
        train_event, train_type, train_time, train_event_loss, train_non_event_loss, train_total_loss = train_epoch(
            model, training_data, optimizer, pred_loss_func, opt, epoch=epoch)
        print('  - (Training)    loglikelihood: {ll: 8.5f}, '
              'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
              ' [loss: {loss:3.3f}] , '
              'elapse: {elapse:3.3f} min '
              .format(ll=train_event, type=train_type, rmse=train_time, elapse=(time.time() - start) / 60, loss=train_total_loss))
        train_event_lls += [train_event]
        train_event_loss_total += [train_event_loss]
        train_non_event_loss_total += [train_non_event_loss]


        with torch.no_grad():
            start = time.time()
            valid_event, valid_type, valid_time, valid_event_loss, valid_non_event_loss, valid_total_loss = eval_epoch(model, validation_data, pred_loss_func, opt)
            print('  - (Validating)  loglikelihood: {ll: 8.5f}, '
                  'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
                  ' [loss: {loss:3.3f}] , '
                  'elapse: {elapse:3.3f} min'
                  .format(ll=valid_event, type=valid_type, rmse=valid_time, elapse=(time.time() - start) / 60, loss=valid_total_loss))

            valid_event_losses += [valid_event]
            valid_pred_losses += [valid_type]
            valid_rmse += [valid_time]
            valid_event_loss_total += [valid_event_loss]
            valid_non_event_loss_total += [valid_non_event_loss]
            valid_total_losses += [valid_total_loss]

            start = time.time()
            test_event, test_type, test_time, test_event_loss, test_non_event_loss, test_total_loss = eval_epoch(model, test_data, pred_loss_func, opt, epoch)
            print('  - (Testing)     loglikelihood: {ll: 8.5f}, '
                  'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
                  ' [loss: {loss:3.3f}] , '
                  'elapse: {elapse:3.3f} min'
                  .format(ll=test_event, type=test_type, rmse=test_time, elapse=(time.time() - start) / 60, loss=test_total_loss))

            test_event_losses += [test_event]
            test_pred_losses += [test_type]
            test_rmse += [test_time]
            test_event_loss_total += [test_event_loss]
            test_non_event_loss_total += [test_non_event_loss]

            print('  - [Info] Maximum ll: {event: 8.5f}, '
                  'Maximum accuracy: {pred: 8.5f}, Minimum RMSE: {rmse: 8.5f}'
                  .format(event=max(test_event_losses), pred=max(test_pred_losses), rmse=min(test_rmse)))
            writer.add_scalar("Loss/val", valid_event, epoch)
            writer.add_scalar("Loss/train", train_event, epoch)
            writer.add_scalar("Loss/test", test_event, epoch)
            writer.add_scalar("Acc/val", valid_type, epoch)
            writer.add_scalar("Acc/train", train_type, epoch)
            writer.add_scalar("Acc/test", test_type, epoch)
            writer.add_scalar("rmse/val", valid_time, epoch)
            writer.add_scalar("rmse/train", train_time, epoch)
            writer.add_scalar("rmse/test", test_time, epoch)
            # logging
            with open(log_name,'a') as f:
                f.write('{epoch}, {ll: 8.5f}, {acc: 8.5f}, {rmse: 8.5f}'
                        ', {train_event_loss: 8.5f}, {train_non_event_loss: 8.5f}'
                        ', {valid_event_loss: 8.5f}, {valid_non_event_loss: 8.5f}'
                        ', {test_event_loss: 8.5f}, {test_non_event_loss: 8.5f}'
                        ', {train_event: 8.5f}, {train_type: 8.5f}, {train_time: 8.5f}'
                        ', {valid_event: 8.5f}, {valid_type: 8.5f}, {valid_time: 8.5f}'
                        ', {test_event: 8.5f}, {test_type: 8.5f}, {test_time:8.5f}\n'
                        .format(epoch=epoch, ll=test_event, acc=test_type, rmse=test_time
                                , train_event_loss=train_event_loss, train_non_event_loss=train_non_event_loss
                                , valid_event_loss=valid_event_loss, valid_non_event_loss=valid_non_event_loss
                                , test_event_loss=test_event_loss, test_non_event_loss=test_non_event_loss
                                , train_event=train_event, train_type=train_type, train_time=train_time
                                , valid_event=valid_event, valid_type=valid_type, valid_time=valid_time
                                , test_event=test_event, test_type=test_type, test_time=test_time))


        # ploting

        # ------ early stopping ---------
        gap = min(valid_total_losses) - valid_total_losses[-1]
        if gap < 0:
            early_step += 1
            pass
        else:
            early_step = 0
            best_state_dict = model.state_dict()

        # if early_step >= opt.patience:
        #     print('Early Stopping')
        #     break

        # scheduler.step()
        # note: use `ReduceLROnPlateau`
        scheduler.step(valid_total_losses[-1])
        lr = optimizer.param_groups[0]["lr"]
        print('  - current lr: {:.8f},            min_lr: {:.8f}'
              .format(lr, opt.min_lr))

        if lr < opt.min_lr:
            print('Early Stopping')
            break




    # --------------- Reload the best model and evaluate
    print('\n\n\n========================================= Evaluation ====================================================\n')
    if best_state_dict is not None:
        print('Load the best model state_dict (val_loss)\n')
        model.load_state_dict(best_state_dict)
    with torch.no_grad():
        train_event, train_type, train_time, train_event_loss, train_non_event_loss, train_total_loss = eval_epoch(model, training_data, pred_loss_func, opt)
        print('  - (Evaluate-Train)    loglikelihood: {ll: 8.5f}, '
              'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
              ' [loss: {loss:3.3f}] , '
              'elapse: {elapse:3.3f} min '
              .format(ll=train_event, type=train_type, rmse=train_time, elapse=(time.time() - start) / 60, loss=train_total_loss))
        train_event_lls += [train_event]
        train_event_loss_total += [train_event_loss]
        train_non_event_loss_total += [train_non_event_loss]

        start = time.time()
        valid_event, valid_type, valid_time, valid_event_loss, valid_non_event_loss, valid_total_loss = eval_epoch(model, validation_data, pred_loss_func, opt)
        print('  - (Evaluate-Val)     loglikelihood: {ll: 8.5f}, '
              'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
              ' [loss: {loss:3.3f}] , '
              'elapse: {elapse:3.3f} min '
              .format(ll=valid_event, type=valid_type, rmse=valid_time, elapse=(time.time() - start) / 60, loss=valid_total_loss))

        valid_event_losses += [valid_event]
        valid_pred_losses += [valid_type]
        valid_rmse += [valid_time]
        valid_event_loss_total += [valid_event_loss]
        valid_non_event_loss_total += [valid_non_event_loss]
        valid_total_losses += [valid_total_loss]

        start = time.time()
        test_event, test_type, test_time, test_event_loss, test_non_event_loss, test_total_loss = eval_epoch(model, test_data, pred_loss_func, opt)
        print('  - (Test)     loglikelihood: {ll: 8.5f}, '
              'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
              ' [loss: {loss:3.3f}] , '
              'elapse: {elapse:3.3f} min '
              .format(ll=test_event, type=test_type, rmse=test_time, elapse=(time.time() - start) / 60, loss=test_total_loss))
        print("\n\n\n\n")

        test_event_losses += [test_event]
        test_pred_losses += [test_type]
        test_rmse += [test_time]
        test_event_loss_total += [test_event_loss]
        test_non_event_loss_total += [test_non_event_loss]

        with open(opt.final_result_log.format(opt.seed),
                'a') as final_f:
            final_f.write('  - (Test)     loglikelihood: {ll: 8.5f}, '
                  'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
                  ' [loss: {loss:3.3f}] , '
                  'elapse: {elapse:3.3f} min '
                  .format(ll=test_event, type=test_type, rmse=test_time, elapse=(time.time() - start) / 60,
                          loss=test_total_loss))
            final_f.write("\n\n\n\n")


    return model


def same_seed(seed):
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    """ Main function. """

    parser = argparse.ArgumentParser()
    parser.add_argument('-root', default='./')
    parser.add_argument('-data', required=True)
    parser.add_argument('-seed', required=True)
    parser.add_argument('-epoch', type=int, default=30)
    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-d_model', type=int, default=64)
    parser.add_argument('-d_rnn', type=int, default=256)
    parser.add_argument('-d_inner_hid', type=int, default=128)
    parser.add_argument('-d_k', type=int, default=16)
    parser.add_argument('-d_v', type=int, default=16)

    parser.add_argument('-n_head', type=int, default=4)
    parser.add_argument('-n_layers', type=int, default=4)


    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-lr', type=float, default=1e-4)

    parser.add_argument('-decay_factor', type=float, default=0.5)
    parser.add_argument('-smooth', type=float, default=0.1)
    parser.add_argument('-log', type=str, default='log.txt')
    parser.add_argument('-n_taylor_terms', type=int, default=5)
    parser.add_argument('-n_samples', type=int, default=100)
    parser.add_argument('-ExpFolder', type=str, default='Exp')
    parser.add_argument('-final_result_log', type=str, default='final_result.txt')

    parser.add_argument('-min_lr', type=float, default=5e-7)
    parser.add_argument('-patience', type=int, default=10)
    parser.add_argument('-ExpHyper', type=str, default='TaylorTermsExp')
    parser.add_argument('-data_name', type=str, default='mimic')
    parser.add_argument('-num_workers', type=int, default=10)
    parser.add_argument('-clamp', type=float, default=5., help="clamping on the attention matrix")

    parser.add_argument('-event_update', action="store_true")
    parser.add_argument('-explicit_time', action="store_true", help="clamping on the attention matrix")
    parser.add_argument('-transf_dec', action="store_true", help="clamping on the attention matrix")
    parser.add_argument('-decaying', action="store_true", help="the influence matrix decay as the time-interval getting large")
    parser.add_argument('-scalar_time', action="store_true", help="The variant using scalar-time (v3)")
    # parser.add_argument('-divided_by_history', action="store_true", help="influence matrix divided by the number of history, --> balance the affect from different history ")
    # parser.add_argument('-transf_dec', action="store_true", help="clamping on the attention matrix")


    # 1: MC, 2: NU
    parser.add_argument('-integral_solver', type=int, default=1)
    parser.add_argument('-dir', type=str, default='./')
    parser.add_argument('-tb_log', type=str, default='./tb_log')
    parser.add_argument('-scale', type=bool, default=True)
    parser.add_argument('-scale_time_loss', type=bool, default=True)
    parser.add_argument('-small_test_fin_num_pred', type=int, default=0)
    parser.add_argument('-attn_log_path', type=str, default='./')

    opt = parser.parse_args()

    # default device is CUD
    opt.device = torch.device('cuda')

    # Creating a datetime object so we can test.
    random_seed = int(opt.seed)

    # setup the log file
    log_name = opt.log.format(random_seed)
    with open(log_name, 'w') as f:
        f.write('Epoch,Log-likelihood,Accuracy,RMSE'
                ',TrainEventLL,TrainNonEventLL,ValEventLL,ValNonEventLL,TestEventLL,TestNonEventLL'
                ',TrainLoss,TrainAcc,TrainRMSE'
                ',ValLoss,ValAcc,ValRMSE'
                ',TestLoss,TestAcc,TestRMSE\n')
    print('[Info] parameters: {}'.format(opt))

    is_exist = os.path.exists(opt.attn_log_path)
    if is_exist == False:
        os.makedirs(opt.attn_log_path)

    """ set random seed """
    same_seed(random_seed)

    """ prepare dataloader """
    # trainloader, validationloader, testloader, num_types, max_length = prepare_dataloader(opt)
    trainloader, validationloader, testloader, num_types = prepare_dataloader(opt)
    """ prepare model """
    model = Transformer(
        num_types=num_types,
        d_model=opt.d_model,
        d_rnn=opt.d_rnn,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        d_k=opt.d_k,
        d_v=opt.d_v,
        num_taylor_terms=opt.n_taylor_terms,
        dropout=opt.dropout,
        clamp=vars(opt).get("clamp", None),
        event_update=vars(opt).get("event_update", None),
        explicit_time=vars(opt).get("explicit_time", None),
        transf_dec=vars(opt).get("transf_dec", False),
        decaying=vars(opt).get("decaying", False)
    )
    # breakpoint()
    model.to(opt.device)
    # breakpoint()
    """ optimizer and scheduler """
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                           opt.lr, betas=(0.9, 0.999), eps=1e-05)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=opt.decay_factor, patience=opt.patience, min_lr=opt.min_lr, threshold=1e-16)

    """ prediction loss function, either cross entropy or label smoothing """
    if opt.smooth > 0:
        pred_loss_func = Utils.LabelSmoothingLoss(opt.smooth, num_types, ignore_index=-1)
    else:
        pred_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[Info] Number of parameters: {}'.format(num_params))

    """ train the model """
    train(model, trainloader, validationloader, testloader, optimizer, scheduler, pred_loss_func, opt, log_name)
    # writer.flush()

if __name__ == '__main__':
    main()
