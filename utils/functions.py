from __future__ import print_function
from __future__ import division
import torch
import os
import pickle

def save_checkpoint(state, is_best, opts, logger=None, epoch=0, criterion="", fold=1):
    """
    Save current model to checkpoints dir
    """
    # --- borrowed from: https://github.com/pytorch/examples/blob/master/imagenet/main.py#L139
    if is_best:
        out_file = os.path.join(
            opts.checkpoints, "".join(["best_under", criterion, "criterion_fold{}.pth".format(fold)])
        )
        torch.save(state, out_file)
        if logger is not None:
            logger.info("Best model saved to {} at epoch {}".format(out_file, str(epoch)))
        else:
            print("Best model saved to {} at epoch {}".format(out_file, str(epoch)))
    else:
        out_file = os.path.join(opts.checkpoints, "checkpoint_fold{}.pth".format(fold))
        torch.save(state, out_file)
        if logger is not None:
            logger.info("Checkpoint saved to {} at epoch {}".format(out_file, str(epoch)))
        else:
            print("Checkpoint saved to {} at epoch {}".format(out_file, str(epoch)))
    return out_file

def save_checkpoint_basic(state, is_best, epoch=0, criterion="", fold=1, checkpoints=""):
    """
    Save current model to checkpoints dir
    """
    # --- borrowed from: https://github.com/pytorch/examples/blob/master/imagenet/main.py#L139
    if is_best:
        out_file = os.path.join(
            checkpoints, "".join(["best_under", criterion, "criterion_fold{}.pth".format(fold)])
        )
        torch.save(state, out_file)

        print("Best model saved to {} at epoch {}".format(out_file, str(epoch)))
    else:
        out_file = os.path.join(checkpoints, "checkpoint_fold{}.pth".format(fold))
        torch.save(state, out_file)
        print("Checkpoint saved to {} at epoch {}".format(out_file, str(epoch)))
    return out_file

def check_inputs(opts, logger):
    """
    check if some inputs are correct
    """
    n_err = 0
    # --- check if input files/folders exists
    if opts.do_train:
        if not (os.path.isdir(opts.tr_data) and os.access(opts.tr_data, os.R_OK)):
            n_err = n_err + 1
            logger.error(
                "Folder {} does not exists or is unreadable".format(opts.tr_data)
            )


    if opts.do_test:
        if not (os.path.isdir(opts.te_data) and os.access(opts.te_data, os.R_OK)):
            n_err = n_err + 1
            logger.error(
                "Folder {} does not exists or is unreadable".format(opts.te_data)
            )


    if opts.do_val:
        if not (os.path.isdir(opts.val_data) and os.access(opts.val_data, os.R_OK)):
            n_err = n_err + 1
            logger.error(
                "Folder {} does not exists or is unreadable".format(opts.val_data)
            )

    # --- if cont_train is defined prev_model must be defined as well
    if opts.cont_train:
        if opts.prev_model == None:
            n_err = n_err + 1
            logger.error("--prev_model must be defined to perform continue training.")
        else:
            if not (
                os.path.isfile(opts.prev_model) and os.access(opts.prev_model, os.R_OK)
            ):
                n_err = n_err + 1
                logger.error(
                    "File {} does not exists or is unreadable".format(opts.prev_model)
                )
        if not opts.do_train:
            logger.warning(
                (
                    "Continue training is defined, but train stage is not. "
                    "Skipping continue..."
                )
            )
    # --- if test,val or prod is performed, train or prev model must be defined
    if opts.do_val:
        if not opts.do_train:
            logger.warning(
                (
                    "Validation data runs only under train stage, but "
                    "no train stage is running. Skipping validation ..."
                )
            )
    if opts.do_test:
        if not (opts.do_train):
            if opts.prev_model == None:
                n_err = n_err + 1
                logger.error(
                    (
                        "There is no model available through training or "
                        "previously trained model. "
                        "Test and Production stages cannot be performed..."
                    )
                )
            else:
                if not (
                    os.path.isfile(opts.prev_model)
                    and os.access(opts.prev_model, os.R_OK)
                ):

                    n_err = n_err + 1
                    logger.error(
                        "File {} does not exists or is unreadable".format(
                            opts.prev_model
                        )
                    )

    return n_err


def check_inputs_graph(opts, logger):
    """
    check if some inputs are correct
    """
    n_err = 0
    # --- check if input files/folders exists
    if not (os.path.isdir(opts.data_path) and os.access(opts.data_path, os.R_OK)):
        n_err = n_err + 1
        logger.error(
            "Folder {} does not exists or is unreadable".format(opts.data_path)
        )



    return n_err

def save_to_file(data, fname):
    with open(fname, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)