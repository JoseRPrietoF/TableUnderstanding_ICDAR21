#!/usr/bin/env python3.6
from __future__ import print_function
from __future__ import division
import time, os, logging, errno
from utils.functions import check_inputs_graph, save_checkpoint
from utils.optparse_graph import Arguments as arguments
from utils import metrics
import numpy as np
import torch, random, glob
from torch_geometric.data import DataLoader
from torch_geometric import transforms as T
import torch.nn.functional as F
# from data import transforms
from models import Graph
from models import operations
# from models import models_p2pala as models_p2pala_2AA, operations
import matplotlib.pyplot as plt
import cv2
from utils import functions
import shutil
from sklearn.model_selection import train_test_split
from utils.metrics import evaluate_graph_IoU

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def tensor_to_numpy(tensor):
    return tensor.cpu().detach().numpy()

def get_all(path, ext="pkl"):
    file_names = glob.glob("{}/*.{}".format(path, ext))
    return file_names

def prepare():
    """
    Logging and arguments
    :return:
    """

    # Logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    # --- keep this logger at DEBUG level, until aguments are processed
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(module)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # --- Get Input Arguments
    in_args = arguments(logger)
    opts = in_args.parse()
    if check_inputs_graph(opts, logger):
        logger.critical("Execution aborted due input errors...")
        exit(1)

    fh = logging.FileHandler(opts.log_file, mode="a")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # --- restore ch logger to INFO
    ch.setLevel(logging.INFO)
    return logger, opts

def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out

def chunkWithFold(seq, opts):
    seq2 = [x.split("/")[-1].split(".")[0] for x in seq]
    seq2 = list(zip(seq,seq2))
    out = []
    fold_files = glob.glob(os.path.join(opts.fold_paths, "fold*txt"))
    for fold_file in fold_files:
        fold_list = []
        f_fold = open(fold_file, "r")
        lines_fold = f_fold.readlines()
        f_fold.close()
        lines_fold = [l.strip().split(".")[0] for l in lines_fold]
        for s_path, s in seq2:
            if s in lines_fold:
                fold_list.append(s_path)
                # print(s_path)
        out.append(fold_list)
        # print(lines_fold[:10])
        # print(seq2[:10])

    return out

def chunkWithMyFold(seq, opts):
    seq2 = [x.split("/")[-1].split(".")[0] for x in seq]
    seq2 = list(zip(seq,seq2))
    out = []
    fold_files = glob.glob(os.path.join(opts.fold_paths, "*fold"))
    for fold_file in fold_files:
        fold_list = []
        f_fold = open(fold_file, "r")
        lines_fold = f_fold.readlines()
        f_fold.close()
        lines_fold = [l.strip().split(".")[0] for l in lines_fold]
        for s_path, s in seq2:
            if s in lines_fold:
                fold_list.append(s_path)
                # print(s_path)
        out.append(fold_list)
        # print(lines_fold[:10])
        # print(seq2[:10])

    return out

def check_splits(splits):
    for i in range(len(splits)):
        s = splits[i]
        for j in range(len(splits)):
            if i==j: continue
            s_j = splits[j]
            for file_s in s:
                if file_s in s_j:
                    print("File {} repeated. In split {} and {}".format(file_s, i, j))

def check_splits_2(tr, te):
    s = 0
    for t in tr:
        if t in te:
            s += 1
            # print("file {} from train is in test partition ".format(t))
    return s

def main():
    global_start = time.time()
    logger, opts = prepare()
    # --- set device
    device = torch.device("cuda:{}".format(opts.gpu) if opts.use_gpu else "cpu")
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    # --- Init torch random
    # --- This two are suposed to be merged in the future, for now keep boot
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    input_channels = opts.input_channels


    # --- Init model variable
    net = None
    bestState = None
    prior = None
    torch.set_default_tensor_type("torch.FloatTensor")
    # --- configure TensorBoard display
    opts.img_size = np.array(opts.img_size, dtype=np.int)
    width, height = np.array(opts.img_size, dtype=np.int)
    logger.info(opts)
    if opts.debug:
        dir_train = os.path.join(opts.work_dir, "debug/train")
        dir_test = os.path.join(opts.work_dir, "debug/test")
        dir_dev = os.path.join(opts.work_dir, "debug/dev")
    opts.segm = True
    # --------------------------------------------------------------------------
    # -----  TRAIN STEP
    # --------------------------------------------------------------------------
    train_start = time.time()
    logger.info("Working on training stage...")
    # --- display is used only on training step
    if not opts.no_display:
        import socket
        from datetime import datetime

        try:
            from tensorboardX import SummaryWriter

            if opts.use_global_log:
                run_dir = opts.use_global_log
            else:
                run_dir = os.path.join(opts.work_dir, "runs")
            log_dir = os.path.join(
                run_dir,
                "".join(
                    [
                        datetime.now().strftime("%b%d_%H-%M-%S"),
                        "_",
                        socket.gethostname(),
                        opts.log_comment,
                    ]
                ),
            )

            writer = SummaryWriter(log_dir=log_dir)
            logger.info("TensorBoard log will be stored at {}".format(log_dir))
            logger.info("run: tensorboard --logdir {}".format(run_dir))
        except:
            logger.warning(
                "tensorboardX is not installed, display logger set to OFF."
            )

            opts.no_display = True
            # --- Build transforms
    transform = T.Compose([
        # T.RandomTranslate(0.01),
        # T.RandomFlip(0), T.RandomFlip(1),
        # T.RandomShear(0.2),
        # T.RandomRotate(20, 0),
        # T.RandomRotate(20, 1),
        # T.Distance(),

    ])
    # transform = None
    # pre_transform = T.Compose([
    #     T.Constant(value=1),
        # T.Distance(),
        # T.KNNGraph(k=6),
    # ])
    pre_transform = None
    # pre_transform = T.Compose([T.KNNGraph(k=6)])
    # --- Get Train Data
    list_files = get_all(opts.data_path)
    if not opts.fold_paths:
        random.shuffle(list_files)
        splits = chunkIt(list_files, 4)
    else:
        if opts.myfolds:
            logger.info("Using our folds")
            splits = chunkWithMyFold(list_files, opts)
        else:
            splits = chunkWithFold(list_files, opts)
    # check_splits(splits)
    [print(len(z)) for z in splits]
    all_splits = []
    for s in splits:
        all_splits.extend(s)
    all_splits = list(set(all_splits))
    logger.info("A total of {} diferent files".format(len(all_splits)))
    acc_splits, p_splits, r_splits, f1_splits = [],[],[],[]
    fp_splits, fr_splits, ff1_splits = [],[],[]
    fp_splits_1, fr_splits_1, ff1_splits_1 = [],[],[]
    Start_time = time.time()
    results_tests = []
    results_loss = []
    res_all_images = []
    res_all_images_1 = []
    for i in range(len(splits)):
        logger.info("{} Fold".format(i))
        # if i < 3:
        #     continue
        list_te = splits[i]
        list_train_val = []
        # for f in list_files:
        #     if f not in list_te:
        #         list_train_val.append(f)
        for j in range(len(splits)):
            if i == j: continue
            list_train_val.extend(splits[j])
        list_train_val = list(set(list_train_val))
        # print(len(list_te))
        # print(len(list_train_val))
        all_used = list_te + list_train_val
        total_rep = check_splits_2(list_train_val, list_te)
        logger.info("Checking splits, {} train {} test, {} unique total files, {} repeated in train".format(len(list_train_val), 
            len(list_te), len(set(all_used)), total_rep ))
        print(len(list_te) + len(list_train_val), len(list_files), len(list_te), len(list_train_val))
        # exit()

        acc,p,r,f1, results_test, results_train, losses, fP, fR, fF, res, fP_1, fR_1, fF_1, res_1 = train(list_train_val, list_te, opts, device, logger, writer=None, transform=transform, pre_transform=pre_transform, show_nn=i==0, nfold=i+1)
        results_loss.append(losses)
        acc_splits.append(acc)
        p_splits.append(p)
        r_splits.append(r)
        f1_splits.append(f1)
        results_tests.extend(results_test)
        if opts.classify != "HEADER":
            fp_splits.append(fP)
            fr_splits.append(fR)
            ff1_splits.append(fF)
            fp_splits_1.append(fP_1)
            fr_splits_1.append(fR_1)
            ff1_splits_1.append(fF_1)
            res_all_images.extend(res)
            res_all_images_1.extend(res_1)
        print(len(results_tests))
        break;
        # exit()
    save_losses(opts, results_loss)
    test_end_time = time.time()
    logger.info("-------------------------------------------------- \n \n")
    logger.info("-> Mean Acc : {}".format(np.mean(acc_splits)))
    logger.info("-> Mean Precision on test : {}".format(np.mean(p_splits)))
    logger.info("-> Mean Recal on test : {}".format(np.mean(r_splits)))
    logger.info("-> Mean F1 on test : {}".format(np.mean(f1_splits)))
    if opts.classify != "HEADER":
        logger.info("#####   IoU and alignment of connected components  #####")
        logger.info("-> Mean Precision IoU th 0.8 on test : {}".format(np.mean(fp_splits)))
        logger.info("-> Mean Recal IoU th 0.8 on test : {}".format(np.mean(fr_splits)))
        logger.info("-> Mean F1 IoU th 0.8 on test : {}".format(np.mean(ff1_splits)))
        logger.info("-> Mean Precision IoU th 1.0 on test : {}".format(np.mean(fp_splits_1)))
        logger.info("-> Mean Recal IoU th 1.0 on test : {}".format(np.mean(fr_splits_1)))
        logger.info("-> Mean F1 IoU th 1.0 on test : {}".format(np.mean(ff1_splits_1)))
        save_results(results_tests, results_train, res_all_images,res_all_images_1, opts.work_dir, logger)
    else:
        save_results_header(results_tests, opts.work_dir, logger)
    logger.info(
        "Total time taken: {}".format(
            test_end_time - Start_time
        )
    )

def save_losses(opts, list_losses):
    file_loss_path = os.path.join(opts.work_dir, "loss.txt")
    file = open(file_loss_path, "w")
    losses = []
    # print(list_losses)
    tr, dev = list_losses[0]
    for i, _ in enumerate(tr):
        losses.append("{} ".format(i))
    for tr, dev in list_losses:
        for i, l in enumerate(tr):
            losses[i] += "{} ".format(l)
        for i, l in enumerate(dev):
            losses[i] += "{} ".format(l)
    for loss in losses:
        file.write(loss+"\n")
    file.close()

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_results_header(results_tests, dir, logger):
    create_dir(dir)
    fname = os.path.join(dir, "results.txt")
    logger.info("Saving results on {}".format(fname))
    f = open(fname, "w")
    f.write("ID LABEL PREDICTION\n")
    for id_line, label, prediction in results_tests:
        f.write("{} {} {}\n".format(id_line, label, prediction))
    f.close()

def save_results(results_tests,results_train, res_all_images, res_all_images_1, dir, logger):
    create_dir(dir)
    fname = os.path.join(dir, "results.txt")
    logger.info("Saving results on {}".format(fname))
    f = open(fname, "w")
    f.write("ID LABEL PREDICTION\n")
    for id_line, label, prediction in results_tests:
        f.write("{} {} {}\n".format(id_line, label, prediction))
    f.close()
    fname = os.path.join(dir, "results_train.txt")
    f = open(fname, "w")
    f.write("ID LABEL PREDICTION\n")
    for id_line, label, prediction in results_train:
        f.write("{} {} {}\n".format(id_line, label, prediction))
    f.close()
    ## ALL images
    fname = os.path.join(dir, "results_IoU_0.8th.txt")
    logger.info("Saving results on {}".format(fname))
    f = open(fname, "w")
    f.write("Graph _nOk, _nErr, _nMiss, _fP, _fR, _fF\n")
    for raw_path,  _nOk, _nErr, _nMiss, _fP, _fR, _fF in res_all_images:
        f.write("{} {} {} {} {} {} {}\n".format(raw_path,  _nOk, _nErr, _nMiss, _fP, _fR, _fF))
    f.close()
    fname = os.path.join(dir, "results_IoU_1.0th.txt")
    logger.info("Saving results on {}".format(fname))
    f = open(fname, "w")
    f.write("Graph _nOk, _nErr, _nMiss, _fP, _fR, _fF\n")
    for raw_path,  _nOk, _nErr, _nMiss, _fP, _fR, _fF in res_all_images_1:
        f.write("{} {} {} {} {} {} {}\n".format(raw_path,  _nOk, _nErr, _nMiss, _fP, _fR, _fF))
    f.close()

def train(list_train_val, list_te, opts, device, logger, writer, transform=None, pre_transform=None, show_nn=False, nfold=1):
    shutil.rmtree(os.path.join(opts.work_dir, "train"), ignore_errors=True)
    shutil.rmtree(os.path.join(opts.work_dir, "test"), ignore_errors=True)
    shutil.rmtree(os.path.join(opts.work_dir, "dev"), ignore_errors=True)

    if opts.classify == "HEADER":
        from data.graph_dataset_header import ABPDataset_Header as ABPDataset
        logger.info("Classifying Headers")
    else:
        from data.graph_dataset import ABPDataset_BIESO as ABPDataset
        logger.info("Classifying edges")

    # if opts.do_val:
    test_dataset = ABPDataset(root=opts.data_path, split="test", flist=list_te, pre_transform=pre_transform,
                                 transform=None,
                                 opts=opts)
    # test_dataloader = ABPDataset_BIESO(root=opts.data_path, split="test", flist=list_tr, transform=transform, opts=opts)
    test_dataset.process()
    print("A total of {} labels in test dataset".format(len(test_dataset.labels)))
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=opts.batch_size,
        shuffle=False,
        # num_workers=opts.num_workers,
        # pin_memory=opts.pin_memory,
    )

    if opts.do_val:
        val_perc = 0.1

        list_tr, list_val = train_test_split(list_train_val, test_size=val_perc, random_state=opts.seed)
        dataset_tr = ABPDataset(root=opts.data_path, split="train", flist=list_tr, transform=transform,
                                pre_transform=pre_transform, opts=opts)
        dataset_tr.process()
        dataset_dev = ABPDataset(root=opts.data_path, split="dev", flist=list_val, transform=transform,
                                 pre_transform=pre_transform, opts=opts)
        dataset_dev.process()

        val_dataloader = DataLoader(
            dataset_dev,
            batch_size=opts.batch_size,
            shuffle=opts.shuffle_data,
            # num_workers=opts.num_workers,
            # pin_memory=opts.pin_memory,
        )
    else:
        dataset_tr = ABPDataset(root=opts.data_path, split="train", flist=list_train_val, transform=transform,
                                pre_transform=pre_transform, opts=opts)
        dataset_tr.process()

    train_dataloader = DataLoader(
        dataset_tr,
        batch_size=opts.batch_size,
        shuffle=opts.shuffle_data,
        # num_workers=opts.num_workers,
        # pin_memory=opts.pin_memory,
    )
    net = Graph.Net(dataset=dataset_tr, opts=opts).to(device)
    ## Loss function
    c_weights = dataset_tr.prob_class
    logger.info("Using {} loss function".format(opts.g_loss))
    if opts.g_loss == "NLL":
        logger.info("Class weight : {}".format(c_weights))
        lossFunc = torch.nn.NLLLoss(reduction="mean", weight=torch.Tensor(c_weights)).to(device)
    elif opts.g_loss == "BCE":
        print(c_weights)
        # c_weights = 1/c_weights
        print(c_weights)
        weight = c_weights[0] / c_weights[1]
        logger.info("Class weight : {}".format(weight))
        # lossFunc = torch.nn.BCEWithLogitsLoss(reduction="mean", weight=torch.tensor(weight)).to(device)
        lossFunc = operations.weighted_binary_cross_entropy
    # lossFunc = torch.nn.CrossEntropyLoss(reduction="mean").to(device)
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=opts.adam_lr, betas=(opts.adam_beta1, opts.adam_beta2)
    )
    if show_nn:
        logger.info(net)
    valOrTrain = "train"
    if opts.do_val:
        valOrTrain = "val"
    model_loaded = False
    path_load = os.path.join(opts.work_dir, "checkpoints", \
        "".join(["best_under", valOrTrain + opts.g_loss, "criterion_fold{}.pth".format(nfold)]))
    
    # if opts.load_model:
    print(path_load)
    if os.path.exists(path_load):
        
        logger.info("Loading model for fold {} from {}".format(nfold, path_load))
        if not os.path.exists(path_load):
            logger.info("Model doesnt exists. Loading avoided.")
        else:
            if opts.use_gpu:
                logger.info("Loading model for GPU")
                checkpoint = torch.load(path_load)
            else:
                logger.info("Loading model for CPU")
                checkpoint = torch.load(
                    path_load, map_location=lambda storage, loc: storage
                )
            net.load_state_dict(checkpoint["net"])
            optimizer.load_state_dict(checkpoint["model_optimizer_state"])
            init_epoch = checkpoint["epoch"]
            best_val = checkpoint["best_loss"]
            best_epoch = init_epoch
            best_tr = best_val
            model_loaded = True
            best_model = path_load
    if not model_loaded:
        logger.info("Initializing new model")
        net.apply(Graph.weights_init_normal)
        init_epoch = 0
        best_epoch = 0
        best_val = np.inf
        best_tr = np.inf
        best_model = ""


    logger.info("Total parameters: {}".format(net.num_params))
    
    loss_tr = []
    loss_dev = []
    logger.info("Starting at epoch {} with best val {}".format(init_epoch, best_val))
    # TODO class imbalance
    epoch = -1
    for epoch in range(init_epoch, opts.epochs):
        epoch_start = time.time()
        epoch_loss = 0
        for batch, sample in enumerate(train_dataloader):
            net.train(True)
            optimizer.zero_grad()
            hyp = net(sample.to(device))
            y = sample.y
            if opts.g_loss == "BCE":
                y = sample.y.type_as(hyp)
            loss = lossFunc(hyp, y)
            # exit()
            loss.backward()
            optimizer.step()
            epoch_loss += loss / sample.y.data.size()[0]
            # if opts.debug:
            #     pass

        epoch_loss = epoch_loss / (batch + 1)  # mean
        loss_tr.append(epoch_loss)
        # print("Epoch {} loss: {}".format(epoch, epoch_loss))
        # --- forward pass val
        val_loss = 0
        if opts.do_val:
            with torch.no_grad():
                net.eval()
                for v_batch, v_sample in enumerate(val_dataloader):
                    # --- set vars to volatile, since bo backward used
                    hyp = net(v_sample.to(device))
                    y = v_sample.y
                    if opts.g_loss == "BCE":
                        y = v_sample.y.type_as(hyp)
                    loss_v = lossFunc(hyp, y)
                    val_loss += loss_v / v_sample.y.data.size()[0]

                    if opts.debug:
                        pass
                val_loss = val_loss / (v_batch + 1)
                loss_dev.append(val_loss)
                # --- Write to Logs
                if not opts.no_display:
                    writer.add_scalar("train/loss", epoch_loss, epoch)
                    writer.add_text(
                        "LOG",
                        "End of epoch {0} of {1} time Taken: {2:.3f} sec".format(
                            str(epoch), str(opts.epochs), time.time() - epoch_start
                        ),
                        epoch,
                    )

                    writer.add_scalar("val/loss", val_loss, epoch)
        if opts.do_val:
            epoch_loss = val_loss
        # --- Save model under val or min loss
        if best_val >= epoch_loss:
            best_epoch = epoch
            state = {
                "net": net.state_dict(),
                "model_optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "best_loss": epoch_loss,

            }
            best_model = save_checkpoint(
                state, True, opts, logger, epoch, criterion=valOrTrain + opts.g_loss, fold=nfold
            )
            logger.info(
                "New best model, from {} to {}".format(best_val, epoch_loss)
            )
            best_val = epoch_loss
        if opts.show_test != -1 and epoch % opts.show_test == 0:
            net.eval()
            res_hyp, res_gt = [], []
            total = 0
            for v_batch, v_sample in enumerate(test_dataloader):
                # f_names = v_sample.fname
                hyp = net(v_sample.to(device))
                if opts.g_loss == "NLL":
                    hyp = hyp[:, 1]
                else:
                    hyp = F.logsigmoid(hyp)
                # _, hyp = torch.max(hyp, dim=1)
                hyp = tensor_to_numpy(hyp)
                y_gt = tensor_to_numpy(v_sample.y)
                total += len(y_gt)
                # print(hyp)
                # print(y_gt)
                res_hyp.append(hyp)
                res_gt.append(y_gt)

            predictions = np.hstack(res_hyp)
            labels = np.hstack(res_gt)

            print("A total of {} labels predicted".format(total))
            acc, p, r, f1 = metrics.eval_graph(labels, predictions)
            logger.info("-" * 10 + "TEST RESULTS SUMMARY --- EPOCH {} ".format(epoch) + "-" * 10)
            logger.info("Fname - Accuracy - Precision - Recall - F1")
            logger.info("Mean Acc on test epoch {} : {}".format(epoch, acc))
            logger.info("Mean Precision on test epoch {}  : {}".format(epoch, p))
            logger.info("Mean Recal on test epoch {} : {}".format(epoch, r))
            logger.info("Mean F1 on test  epoch {}  : {}".format(epoch, f1))
            if opts.classify == "HEADER":
                pass
            else:
                results_test = list(zip(test_dataset.ids, labels, predictions))
                fP, fR, fF, res = evaluate_graph_IoU(list_te, results_test, th=0.8, type_=opts.conjugate, pruned=opts.do_prune)
                fP_1, fR_1, fF_1, res_1 = evaluate_graph_IoU(list_te, results_test, th=1.0, type_=opts.conjugate, pruned=opts.do_prune)

                logger.info("-" * 10 + "TEST RESULTS SUMMARY --- EPOCH {} ".format(epoch) + "-" * 10)
                logger.info("Fname - Accuracy - Precision - Recall - F1")
                logger.info("Mean Acc on test epoch {} : {}".format(epoch, acc))
                logger.info("Mean Precision on test epoch {}  : {}".format(epoch, p))
                logger.info("Mean Recal on test epoch {} : {}".format(epoch, r))
                logger.info("Mean F1 on test  epoch {}  : {}".format(epoch, f1))
                logger.info("#####   IoU and alignment of connected components  #####")
                logger.info("Mean Precision IoU on test : {}".format(fP))
                logger.info("Mean Recal IoU on test : {}".format(fR))
                logger.info("Mean F1 IoU on test : {}".format(fF))
                logger.info("Mean Precision IoU th 1.0 on test : {}".format(fP_1))
                logger.info("Mean Recal IoU th 1.0 on test : {}".format(fR_1))
                logger.info("Mean F1 IoU th 1.0 on test : {}".format(fF_1))
            

    # logger.info(
    #     "Training stage done. Total time taken: {}".format(time.time() - train_start)
    # )
    part = "validation"
    if not opts.do_val:
        val_dataloader = train_dataloader
        part = "train"
    # ---- Train is done, next is to save validation inference
    logger.info("Working on {} inference...".format(part))
    res_path = os.path.join(opts.work_dir, "results", "val")
    try:
        os.makedirs(os.path.join(res_path, "page"))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(
                os.path.join(res_path, "page")
        ):
            pass
        else:
            raise
    # --- Set model to eval, to perform inference step
    with torch.no_grad():
        if best_epoch != epoch and epoch != -1:
            # --- load best model for inference
            if opts.use_gpu:
                checkpoint = torch.load(best_model)
                net.load_state_dict(checkpoint["net"])
            else:
                checkpoint = torch.load(
                    best_model, map_location=lambda storage, loc: storage
                )
                net.load_state_dict(checkpoint["net"])

    res_hyp, res_gt = [], []
    for v_batch, v_sample in enumerate(val_dataloader):
        # f_names = v_sample.fname
        hyp = net(v_sample.to(device))
        if opts.g_loss == "NLL":
            hyp = hyp[:, 1]
        else:
            hyp = F.logsigmoid(hyp)
        # _, hyp = torch.max(hyp, dim=1)
        hyp = tensor_to_numpy(hyp)
        gt = tensor_to_numpy(v_sample.y)
        res_hyp.append(hyp)
        res_gt.append(gt)

    predictions = np.hstack(res_hyp)
    labels = np.hstack(res_gt)

    s = 0
    for i in range(len(predictions)):
        # print("{} - {} - {}".format(predictions[i], labels[i], predictions[i]==labels[i]))
        s += predictions[i]==labels[i]
    acc, p, r, f1 = metrics.eval_graph(labels, predictions)

    logger.info("-" * 10 + "{} RESULTS SUMMARY".format(part) + "-" * 10)
    logger.info("Fname - Accuracy - Precision - Recall - F1")
    # for fname, acc, p, r, f1 in per_img:
    #     logger.info(
    #         "{} - {} - {} - {} - {} ".format(fname, acc, p, r, f1))
    logger.info("Mean Acc on {} : {}".format(part, np.mean(acc)))
    logger.info("Mean Precision on {} : {}".format(part, np.mean(p)))
    logger.info("Mean Recal on {} : {}".format(part, np.mean(r)))
    logger.info("Mean F1 on {} : {}".format(part, np.mean(f1)))
    logger.info("s : {} / {}".format(s, len(predictions)) )

    # --------------------------------------------------------------------------
    # ---    TRAIN INFERENCE
    # --------------------------------------------------------------------------
    with torch.no_grad():
        logger.info("Working on train inference...")
        # --- get test data
        test_start_time = time.time()

        res_hyp_tr, res_gt_tr = [], []
        net.eval()
        total = 0
        for v_batch, v_sample in enumerate(train_dataloader):
            # f_names = v_sample.fname
            hyp = net(v_sample.to(device))
            if opts.g_loss == "NLL":
                hyp = hyp[:,1]
            else:
                hyp = F.logsigmoid(hyp)
            # _, hyp = torch.max(hyp, dim=1)
            hyp = tensor_to_numpy(hyp)
            y_gt = tensor_to_numpy(v_sample.y)
            total += len(y_gt)
            # print(hyp)
            # print(y_gt)
            res_hyp_tr.append(hyp)
            res_gt_tr.append(y_gt)

        predictions = np.hstack(res_hyp_tr)
        labels = np.hstack(res_gt_tr)
        print("A total of {} labels predicted".format(total))

        results_train = list(zip(dataset_tr.ids, labels, predictions))
        

        acc, p, r, f1 = metrics.eval_graph(labels, predictions)

        test_end_time = time.time()

        logger.info("-" * 10 + "TRAIN RESULTS SUMMARY" + "-" * 10)
        logger.info("Fname - Accuracy - Precision - Recall - F1")
        logger.info("Mean Acc on train : {}".format(acc))
        logger.info("Mean Precision on train : {}".format(p))
        logger.info("Mean Recal on train : {}".format(r))
        logger.info("Mean F1 on train : {}".format(f1))
        if opts.classify == "HEADER":
            fP, fR, fF, res, fP_1, fR_1, fF_1, res_1 = 0,0,0,0,0,0,0,0
        else:
            fP, fR, fF, res = evaluate_graph_IoU(list_train_val, results_train, th=0.8, type_=opts.conjugate, pruned=opts.do_prune)
            fP_1, fR_1, fF_1, res_1 = evaluate_graph_IoU(list_train_val, results_train, th=1.0, type_=opts.conjugate, pruned=opts.do_prune)
            logger.info("#####   IoU and alignment of connected components  #####")
            logger.info("Mean Precision IoU th 0.8 on train : {}".format(fP))
            logger.info("Mean Recal IoU th 0.8 on train : {}".format(fR))
            logger.info("Mean F1 IoU th 0.8 on train : {}".format(fF))
            logger.info("Mean Precision IoU th 1.0 on train : {}".format(fP_1))
            logger.info("Mean Recal IoU th 1.0 on train : {}".format(fR_1))
            logger.info("Mean F1 IoU th 1.0 on train : {}".format(fF_1))

    # --------------------------------------------------------------------------
    # ---    TEST INFERENCE
    # --------------------------------------------------------------------------
    with torch.no_grad():
        logger.info("Working on test inference...")
        # --- get test data
        test_start_time = time.time()

        res_hyp, res_gt = [], []
        net.eval()
        total = 0
        for v_batch, v_sample in enumerate(test_dataloader):
            # f_names = v_sample.fname
            hyp = net(v_sample.to(device))
            if opts.g_loss == "NLL":
                hyp = hyp[:,1]
            else:
                hyp = F.logsigmoid(hyp)
            # _, hyp = torch.max(hyp, dim=1)
            hyp = tensor_to_numpy(hyp)
            y_gt = tensor_to_numpy(v_sample.y)
            total += len(y_gt)
            # print(hyp)
            # print(y_gt)
            res_hyp.append(hyp)
            res_gt.append(y_gt)

        predictions = np.hstack(res_hyp)
        labels = np.hstack(res_gt)
        print("A total of {} labels predicted".format(total))

        results_test = list(zip(test_dataset.ids, labels, predictions))
        

        acc, p, r, f1 = metrics.eval_graph(labels, predictions)

        test_end_time = time.time()

        logger.info("-" * 10 + "TEST RESULTS SUMMARY" + "-" * 10)
        logger.info("Fname - Accuracy - Precision - Recall - F1")
        logger.info("Mean Acc on test : {}".format(acc))
        logger.info("Mean Precision on test : {}".format(p))
        logger.info("Mean Recal on test : {}".format(r))
        logger.info("Mean F1 on test : {}".format(f1))
        if opts.classify == "HEADER":
            fP, fR, fF, res, fP_1, fR_1, fF_1, res_1 = 0,0,0,0,0,0,0,0
        else:
            fP, fR, fF, res = evaluate_graph_IoU(list_te, results_test, th=0.8, type_=opts.conjugate, pruned=opts.do_prune)
            fP_1, fR_1, fF_1, res_1 = evaluate_graph_IoU(list_te, results_test, th=1.0, type_=opts.conjugate, pruned=opts.do_prune)
            logger.info("#####   IoU and alignment of connected components  #####")
            logger.info("Mean Precision IoU th 0.8 on test : {}".format(fP))
            logger.info("Mean Recal IoU th 0.8 on test : {}".format(fR))
            logger.info("Mean F1 IoU th 0.8 on test : {}".format(fF))
            logger.info("Mean Precision IoU th 1.0 on test : {}".format(fP_1))
            logger.info("Mean Recal IoU th 1.0 on test : {}".format(fR_1))
            logger.info("Mean F1 IoU th 1.0 on test : {}".format(fF_1))


        logger.info(
            "Test stage done. total time taken: {}".format(
                test_end_time - test_start_time
            )
        )
        logger.info(
            "Average time per page: {}".format(
                (test_end_time - test_start_time) / len(test_dataloader)
            )
        )
        shutil.rmtree(os.path.join(opts.work_dir, "train"), ignore_errors=True)
        shutil.rmtree(os.path.join(opts.work_dir, "test"), ignore_errors=True)
        shutil.rmtree(os.path.join(opts.work_dir, "dev"), ignore_errors=True)

        return acc, p, r, f1, results_test, results_train, (loss_tr, loss_dev), fP, fR, fF, res, fP_1, fR_1, fF_1, res_1


if __name__ == "__main__":
    main()
    # path_preprocessed = "/data/READ_ABP_TABLE/preprocess_111/rows_cols_1024_768/"
    # dataset_tr = TableRowColumnDataset(path=path_preprocessed, transform=None, opts=None)
    # train_dataloader = DataLoader(
    #     dataset_tr,
    #     batch_size=1,
    #     shuffle=True,
    #     num_workers=1,
    # )
    # for i in train_dataloader:
    #     print(i['img'].size())
    #     print(i['id'])
