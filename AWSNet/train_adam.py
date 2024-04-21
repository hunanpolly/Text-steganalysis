from __future__ import print_function
import os
import math
import sys
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from pytorch_pretrained_bert.optimization import BertAdam
import time

cuda = True
seed = 8


torch.manual_seed(seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
if cuda:
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子

def train(source1_iter, target_iter, test_iter, model, args):
    if args.cuda:
        model.cuda()
    step = 0
    correct = 0
    acc = 0
    min_loss = 10000
    last_step = 0
    ticks = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    #w = open('log/' + str(ticks) + '.txt', 'a+')


    optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad,model.parameters()), args.lr, weight_decay=1e-6)
    model.train()
    for epoch in range(1, args.epochs + 1):
        #w = open('log/' + str(ticks) + '.txt', 'a+')
        print('\n--------training epochs: {}-----------'.format(epoch))
        model.train()

        for i in range(len(source1_iter)):
            step += 1
            # --------------------先处理第一个源域-----------------------------
            batch = next(source1_iter)
            source_data, source_label = batch[0], batch[1]
            batch_t = next(target_iter)
            target_data, __ = batch_t[0], batch_t[1]  # 目标域无标签数据

            optimizer.zero_grad()  # 将梯度初始化为零（因为一个batch的loss关于weight的导数是所有sample的loss关于weight的导数的累加和）
            cls_loss, mmd_loss = model(source_data, target_data, source_label, mark=1)  # 网络返回三个loss值
            gamma = 2 / (1 + math.exp(-10 * (step) / (args.epochs*len(source1_iter)))) - 1
            #gamma = step / args.epochs*len(source1_iter)
            loss = cls_loss + gamma * (mmd_loss)
            loss.backward()  # 处理第一个源域。先更新一波参数（也更新了第二个源域对应提取器分类器的参数，因为disc loss里有他们的参数）
            optimizer.step()

            if step % args.log_interval == 0:
                print('Train source1 batch th/epoch th: {} / {}\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}'.format(
                    step, epoch, loss.item(), cls_loss.item(), mmd_loss.item()))
                result_file = os.path.join(args.save_dir, 'loss.txt')
                with open(result_file, 'a', errors='ignore') as f:
                    f.write('The testing Loss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f} \n'.format(loss.item(), cls_loss.item(), mmd_loss.item()))


            # -----------------------------test----------------------------
            if step % args.test_interval == 0:
                t_acc, _ = test(test_iter, model, args)  # 只做测试 不进行优化 此时时有标签的T数据（对所有test target测试）

                if loss < min_loss:
                    min_loss = loss
                    last_step = step
                    if args.save_best:
                        save(model, args.save_dir, 'best', step)


                print("\nsource_1: twitter, source_2:movie to news %s max acc:" % acc, "\n")
                print('------------------------------------------------------------------------------')
                model.train()

            # ----------------------------save-----------------------
            if step % args.save_interval == 0:
                save(model, args.save_dir, 'snapshot', step % 2000)

        #w.close()


def test(test_iter, model, args):
    model.eval()
    test_loss = 0
    correct = 0
    cnt = 0
    batch_num = 0
    pred_total = None
    test_label_total = []
    for batch in test_iter:
        if cnt >= len(test_iter):
            break
        cnt += 1

        test_data, test_label = batch[0], batch[1]

        with torch.no_grad():
            pred = model(test_data, mark=0)  # num_class个概率



        if pred_total is None:
            pred_total = torch.nn.functional.softmax(pred, dim=1)
        else:
            pred_total = torch.cat([pred_total, torch.nn.functional.softmax(pred, dim=1)], 0)
        test_label_total.extend(test_label.tolist())

        loss = F.nll_loss(F.log_softmax(pred, dim=1), test_label)
        test_loss += loss.item()
        batch_num += 1
        #pred = pred.data.max(1)[1]  # 平均预测结果（概率最大的那个取为1）
        correct += (torch.max(pred, 1)[1].view(test_label.size()).data \
                                 == test_label.data).sum()

    test_loss /= batch_num        
    len_test = 2000
    acc_ = correct.item()/len_test

    if not args.test:
        #cmd
        print('on Target, ', 'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len_test, 100. * acc_))
        result_file = os.path.join(args.save_dir, 'Tloss.txt')
        with open(result_file, 'a', errors='ignore') as f:
            f.write('The testing Loss: {:.4f}\tACCURACY: {:.2f} \n'.format(test_loss, 100. * acc_))

        return acc_, test_loss
    else:
        from sklearn import metrics
        predictions = torch.max(pred_total, 1)[1].cpu().detach().numpy()
        labels = np.array(test_label_total)
        accuracy = metrics.accuracy_score(labels, predictions)
        precious = metrics.precision_score(labels, predictions)
        recall = metrics.recall_score(labels, predictions)
        F1_score = metrics.f1_score(labels, predictions, average='weighted')
        TN = sum((predictions == 0) & (labels == 0))
        TP = sum((predictions == 1) & (labels == 1))
        FN = sum((predictions == 0) & (labels == 1))
        FP = sum((predictions == 1) & (labels == 0))
        #print('\nTesting - loss:{:.6f} acc:{:.4f}({}/{})'.format(loss, accuracy, corrects, size))
        result_file = os.path.join(args.save_dir, 'result.txt')
        with open(result_file, 'a', errors='ignore') as f:
            f.write('The testing accuracy: {:.4f} \n'.format(accuracy))
            f.write('The testing precious: {:.4f} \n'.format(precious))
            f.write('The testing recall: {:.4f} \n'.format(recall))
            f.write('The testing F1_score: {:.4f} \n'.format(F1_score))
            f.write('The testing TN: {} \n'.format(TN))
            f.write('The testing TP: {} \n'.format(TP))
            f.write('The testing FN: {} \n'.format(FN))
            f.write('The testing FP: {} \n\n'.format(FP))
        return accuracy, recall, precious, F1_score


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)

