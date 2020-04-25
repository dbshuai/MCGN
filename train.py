import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import utils
from model.imn import IMN,NewImn


def get_prob(epoch_count):
    prob = 5 / (5 + np.exp(epoch_count / 5))
    return prob


def var_batch(*args):
    package = []
    for tensor in args:
        package.append(tensor.cuda())
    return package


def train(model, optimizer, dataset_loader, gold_prob):
    model.train()
    total_loss = 0
    for step, (sent_index, ae_tag, as_tag, op_label_input,ap_label_input, mask, length) in enumerate(dataset_loader):
        p_gold_op = gold_prob.expand_as(mask)
        [sent_index, ae_tag, as_tag, op_label_input,ap_label_input, mask, p_gold_op,
         length] = var_batch(sent_index,
                             ae_tag, as_tag,
                             op_label_input,
                             ap_label_input,
                             mask,
                             p_gold_op,
                             length)
        model.zero_grad()
        aspect_probs, sentiment_probs = model(sent_index,op_label_input,ap_label_input, p_gold_op, mask)
        length_sorted, sort = torch.sort(length, dim=0, descending=True)
        aspect_probs_sorted = torch.index_select(aspect_probs, dim=0, index=sort)
        sentiment_probs_sorted = torch.index_select(sentiment_probs, dim=0, index=sort)
        ae_tag_sorted = torch.index_select(ae_tag, dim=0, index=sort)
        as_tag_sorted = torch.index_select(as_tag, dim=0, index=sort)
        aspect_probs = torch.nn.utils.rnn.pack_padded_sequence(aspect_probs_sorted, length_sorted, batch_first=True)
        sentiment_probs = torch.nn.utils.rnn.pack_padded_sequence(sentiment_probs_sorted, length_sorted,
                                                                  batch_first=True)
        ae_tag = torch.nn.utils.rnn.pack_padded_sequence(ae_tag_sorted, length_sorted, batch_first=True)
        as_tag = torch.nn.utils.rnn.pack_padded_sequence(as_tag_sorted, length_sorted, batch_first=True)
        loss1 = torch.nn.functional.nll_loss(aspect_probs.data, ae_tag.data)
        loss2 = torch.nn.functional.nll_loss(sentiment_probs.data, as_tag.data)
        loss = loss1 + loss2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss
    return total_loss


def test(model, dataset_loader, gold_prob, name="test"):
    ae_tag_total = []
    as_tag_total = []
    aspect_probs_total = []
    sentiment_probs_total = []
    length_total = []
    model.eval()
    tit = time.time()
    n = 0
    for step, (sent_index, ae_tag, as_tag, op_label_input,ap_label_input, mask, length) in enumerate(dataset_loader):
        p_gold_op = gold_prob.expand_as(mask)
        [sent_index, op_label_input,ap_label_input, mask, p_gold_op] = var_batch(sent_index, op_label_input,ap_label_input, mask, p_gold_op)

        aspect_probs, sentiment_probs = model(sent_index,op_label_input, ap_label_input, p_gold_op, mask)
        _, aspect_probs = torch.max(aspect_probs, dim=-1)
        _, sentiment_probs = torch.max(sentiment_probs, dim=-1)
        aspect_probs = aspect_probs.cpu().detach().numpy().tolist()
        sentiment_probs = sentiment_probs.cpu().detach().numpy().tolist()
        ae_tag = ae_tag.cpu().numpy().tolist()
        as_tag = as_tag.cpu().numpy().tolist()
        length = length.cpu().numpy().tolist()
        ae_tag_total.extend(ae_tag)
        as_tag_total.extend(as_tag)
        aspect_probs_total.extend(aspect_probs)
        sentiment_probs_total.extend(sentiment_probs)
        length_total.extend(length)
    f1_aspect, f1_opinion, f1_ABSA = utils.eval(ae_tag_total, aspect_probs_total, as_tag_total, sentiment_probs_total,
                                                length_total)
    print("  Predicting {:d} examples using {:5.4f} seconds".format(n, time.time() - tit))
    return f1_aspect, f1_opinion, f1_ABSA


def main(args):
    # define location to save the model
    if args.save == "__":
        args.save = "save/IMN_%d_%d" % \
                    (args.lr, args.batch_size)
    ''' make sure the folder to save models exist '''
    if not os.path.exists(args.save):
        os.mkdir(args.save)

    in_dir = "../data/pkl/IMN/"

    datasets = utils.pkl_load(in_dir + "features.pkl")
    sent_train_index = datasets["train_sent"]["index"]
    sent_test_index = datasets["test_sent"]["index"]

    ae_tag_train = datasets["train_sent"]["target_opinion"]
    as_tag_train = datasets["train_sent"]["target_polarity"]

    ae_tag_test = datasets["test_sent"]["target_opinion"]
    as_tag_test = datasets["test_sent"]["target_polarity"]

    op_label_input_train = datasets["train_sent"]["opinion_ex"]
    op_label_input_test = datasets["test_sent"]["opinion_ex"]

    ap_label_input_train = datasets["train_sent"]["aspect_ex"]
    ap_label_input_test = datasets["test_sent"]["aspect_ex"]

    train_batch = args.batch_size
    test_batch = args.batch_size

    train_sent_set = utils.SentDataset(sent_train_index, ae_tag_train, as_tag_train, op_label_input_train,
                                       ap_label_input_train,args.max_len)
    test_set = utils.SentDataset(sent_test_index,
                                 ae_tag_test,
                                 as_tag_test,
                                 op_label_input_test,
                                 ap_label_input_test,
                                 args.max_len)
    test_set_loader = DataLoader(dataset=test_set,
                                 batch_size=test_batch,
                                 shuffle=False)
    train_sent_loader = DataLoader(dataset=train_sent_set,
                                   batch_size=train_batch,
                                   shuffle=True)
    general_embeddings = utils.pkl_load(in_dir + "general_embeddings.pkl")
    domain_embeddings = utils.pkl_load(in_dir + "domain_embeddings.pkl")
    general_embeddings = torch.from_numpy(general_embeddings).float()
    domain_embeddings = torch.from_numpy(domain_embeddings).float()
    model = NewImn(general_embeddings,
                domain_embeddings,
                ae_nums=args.ae_nums,
                as_nums=args.as_nums,
                ds_nums=args.ds_nums,
                iters=args.iters,
                dropout=args.dropout,
                use_transission=args.use_transission)
    weight_p, bias_p = [], []
    for name, p in model.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]
    optimizer = torch.optim.Adam([
        {'params': weight_p, 'weight_decay': 0},
        {'params': bias_p, 'weight_decay': 0}
    ], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95, last_epoch=-1)
    model.cuda()
    f_aspect_best, f_opinion_best, f_absa_best = -np.inf, -np.inf, -np.inf
    tic = time.time()
    print("-----------------------------", args.epochs, len(train_sent_set), args.batch_size)
    for i in range(args.epochs):
        gold_prob = get_prob(i)
        rnd = np.random.uniform()
        # as epoch increasing, the probability of using gold opinion label descreases.
        if rnd < gold_prob:
            p_gold_op_train = np.ones((1, args.max_len))
        else:
            p_gold_op_train = np.zeros((1, args.max_len))
        # p_gold_op_train = np.zeros((1, args.max_len))
        p_gold_op_train = torch.from_numpy(p_gold_op_train).float()
        print("--------------\nEpoch %d begins!" % (i))
        loss = train(model, optimizer, train_sent_loader, p_gold_op_train)
        print("loss=", loss)
        print("  using %.5f seconds" % (time.time() - tic))
        tic = time.time()
        print("\n  Begin to predict the results on Validation")
        p_gold_op_test = torch.from_numpy(np.zeros((1, args.max_len))).float()
        f_aspect, f_opinion, f_absa = test(model, test_set_loader, p_gold_op_test, name="train")
        print("  ---%f   %f   %f---"%(f_aspect,f_opinion,f_absa))
        print("  ----Old best aspect f1 score on test is %f" % f_aspect_best)
        print("  ----Old best opinion f1 score on test is %f" % f_opinion_best)
        print("  ----Old best ABSA f1 score on test is %f" % f_absa_best)
        if f_aspect > f_aspect_best:
            print("  ----New best aspect f1 score on test is %f" % f_aspect)
            f_aspect_best = f_aspect
        if f_opinion > f_opinion_best:
            print("  ----New best opinion f1 score on test is %f" % f_opinion)
            f_opinion_best = f_opinion
        if f_absa > f_absa_best:
            print("  ----New best ABSA f1 score on test is %f" % f_absa)
            f_absa_best = f_absa
            with open(args.save + "/model.pt", 'wb') as to_save:
                torch.save(model, to_save)
        scheduler.step()
        print("lr=", optimizer.param_groups[0]['lr'])
    print("best ABSA f1 score on test is %f" % f_absa_best)
    print("best Aspect f1 score on test is %f" % f_aspect_best)
    print("best Opinion f1 score on test is %f" % f_opinion_best)
    log = f"{args.iters}.log"
    with open(log,"w") as f:
        f.write("best ABSA f1 score on test is %f" % f_absa_best)
        f.write("best Aspect f1 score on test is %f" % f_aspect_best)
        f.write("best Opinion f1 score on test is %f" % f_opinion_best)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ''' load data and save model'''
    parser.add_argument("--save", type=str, default="__",
                        help="path to save the model")
    parser.add_argument("--dataset", type=str, default="laptops")

    ''' model parameters '''
    parser.add_argument("--max_len", type=int, default=83)
    parser.add_argument("--ae_nums", type=int, default=5,
                        help="aspect tag classes")
    parser.add_argument("--as_nums", type=int, default=5,
                        help="polarity tag classes")
    parser.add_argument("--dd_nums", type=int, default=1,
                        help="doc domain classes")
    parser.add_argument("--ds_nums", type=int, default=3)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--use_transission", type=bool, default=True)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--epochs", type=int, default=50,
                        help="number of train epoch")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="batch size")
    parser.add_argument("--dropout", type=float, default=0.55,
                        help="dropout rate")
    parser.add_argument("--seed", type=int, default=123456,
                        help="random seed for reproduction")
    my_args = parser.parse_args()
    torch.manual_seed(my_args.seed)
    np.random.seed(my_args.seed)
    main(my_args)
