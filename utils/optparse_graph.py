from __future__ import print_function
from __future__ import division

import numpy as np
from collections import OrderedDict
import argparse
import os

# from math import log
import multiprocessing
import logging

# from evalTools.metrics import levenshtein


class Arguments(object):
    """
    """

    def __init__(self, logger=None):
        """
        """
        self.logger = logger or logging.getLogger(__name__)
        parser_description = """
        NN Implentation for Layout Analysis
        """
        n_cpus = multiprocessing.cpu_count()

        self.parser = argparse.ArgumentParser(
            description=parser_description,
            fromfile_prefix_chars="@",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        # ----------------------------------------------------------------------
        # ----- Define general parameters
        # ----------------------------------------------------------------------
        general = self.parser.add_argument_group("General Parameters")
        general.add_argument(
            "--model",
            default="EdgeConv",
            type=str,
            help="""name of the model to use. gat | EdgeFeatsConv | EdgeFeatsConvMult | EdgeConv | EdgeUpdateConv | NodeFeatsConv | NodeFeatsConvv2 | NodeFeatsConvv3""",
        )
        general.add_argument(
            "--root_weight",
            default=False,
            action="store_true",
            help="Use root weight",
        )
     
        general.add_argument(
            "--config", default=None, type=str, help="Use this configuration file"
        ) # not tested
        general.add_argument(
            "--exp_name",
            default="table_exp",
            type=str,
            help="""Name of the experiment. Models and data 
                                       will be stored into a folder under this name""",
        )
        general.add_argument(
            "--work_dir", default="./work/", type=str, help="Where to place output data"
        )

        general.add_argument(
            "--log_level",
            default="INFO",
            type=str,
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="Logging level",
        )
        general.add_argument(
            "--conjugate",
            default="NO",
            type=str,
            choices=["NO", "ROW", "COL", "CELL", "ALL", "CR"],
            help="Logging level",
        )
        general.add_argument(
            "--do_prune",
            default=False,
            type=bool,
            help="Compute prior distribution over classes",
        )
        general.add_argument(
            "--results_prune_te",
            default='',
            type=str,
            help="Layers for the FF or RNN network",
        )
        general.add_argument(
            "--results_prune_tr",
            default='',
            type=str,
            help="Layers for the FF or RNN network",
        )
        general.add_argument(
            "--layers",
            default='6,12,18',
            type=str,
            help="Layers for the FF or RNN network",
        )
        general.add_argument(
            "--layers_MLP",
            default='6,12,18',
            type=str,
            help="Layers for the FF or RNN network",
        )
        general.add_argument(
            "--classify",
            default="CELLS",
            type=str,
            choices=["HEADER", "CELLS", "EDGES"],
            help="Logging level",
        )
        # general.add_argument('--baseline_evaluator', default=baseline_evaluator_cmd,
        #                     type=str, help='Command to evaluate baselines')
        general.add_argument(
            "--num_workers",
            default=n_cpus,
            type=int,
            help="""Number of workers used to proces 
                                  input data. If not provided all available
                                  CPUs will be used.
                                  """,
        )
        general.add_argument(
            "--show_test",
            default=-1,
            type=int,
            help="""Do test eval every X epochs. -1 number for no test eval
                                          """,
        )
        general.add_argument(
            "--gpu",
            default=0,
            type=int,
            help=(
                "GPU id. Use -1 to disable. "
            ),
        )
        general.add_argument(
            "--seed",
            default=0,
            type=int,
            help="Set manual seed for generating random numbers",
        )
        general.add_argument(
            "--no_display",
            default=False,
            action="store_true",
            help="Do not display data on TensorBoard",
        )
        general.add_argument(
            "--only_preprocess",
            default=False,
            action="store_true",
            help="only_preprocess",
        )
        general.add_argument(
            "--fasttext",
            default=False,
            action="store_true",
            help="fasttext",
        )
        general.add_argument(
            "--text_info",
            default=False,
            action="store_true",
            help="text_info",
        )
        general.add_argument(
            "--text_length",
            default=False,
            action="store_true",
            help="text_length",
        )
        general.add_argument(
            "--not_preprocessed",
            default=False,
            action="store_true",
            help="not_preprocessed",
        )
        general.add_argument(
            "--img_feats",
            default=False,
            action="store_true",
            help="img_feats",
        )
        general.add_argument(
            "--use_global_log",
            default="",
            type=str,
            help="Save TensorBoard log on this folder instead default",
        )
        general.add_argument(
            "--activation",
            default="ReLU",
            type=str,
            help="ReLU or Mish",
        )
        general.add_argument(
            "--log_comment",
            default="",
            type=str,
            help="Add this commaent to TensorBoard logs name",
        )

        # ----------------------------------------------------------------------
        # ----- Define processing data parameters
        # ----------------------------------------------------------------------
        data = self.parser.add_argument_group("Data Related Parameters")
        data.add_argument(
            "--img_size",
            default=[1024, 768],
            nargs=2,
            type=self._check_to_int_array,
            help="Scale images to this size. Format --img_size H W",
        )


        # ----------------------------------------------------------------------
        # ----- Define dataloader parameters
        # ----------------------------------------------------------------------
        loader = self.parser.add_argument_group("Data Loader Parameters")
        loader.add_argument(
            "--batch_size", default=6, type=int, help="Number of images per mini-batch"
        )
        l_meg1 = loader.add_mutually_exclusive_group(required=False)
        l_meg1.add_argument(
            "--shuffle_data",
            dest="shuffle_data",
            action="store_true",
            help="Suffle data during training",
        )
        l_meg1.add_argument(
            "--no-shuffle_data",
            dest="shuffle_data",
            action="store_false",
            help="Do not suffle data during training",
        )
        l_meg1.set_defaults(shuffle_data=True)
        l_meg2 = loader.add_mutually_exclusive_group(required=False)
        l_meg2.add_argument(
            "--pin_memory",
            dest="pin_memory",
            action="store_true",
            help="Pin memory before send to GPU",
        )
        l_meg2.add_argument(
            "--no-pin_memory",
            dest="pin_memory",
            action="store_false",
            help="Pin memory before send to GPU",
        )
        l_meg2.set_defaults(pin_memory=True)
        l_meg3 = loader.add_mutually_exclusive_group(required=False)
        l_meg3.add_argument(
            "--flip_img",
            dest="flip_img",
            action="store_true",
            help="Randomly flip images during training",
        )
        l_meg3.add_argument(
            "--no-flip_img",
            dest="flip_img",
            action="store_false",
            help="Do not randomly flip images during training",
        )
        l_meg3.set_defaults(flip_img=False)

        elastic_def = loader.add_mutually_exclusive_group(required=False)
        elastic_def.add_argument(
            "--elastic_def",
            dest="elastic_def",
            action="store_true",
            help="Use elastic deformation during training",
        )
        elastic_def.add_argument(
            "--no-elastic_def",
            dest="elastic_def",
            action="store_false",
            help="Do not Use elastic deformation during training",
        )
        elastic_def.set_defaults(elastic_def=True)

        loader.add_argument(
            "--e_alpha",
            default=0.045,
            type=float,
            help="alpha value for elastic deformations",
        )
        loader.add_argument(
            "--e_stdv",
            default=4,
            type=float,
            help="std dev value for elastic deformations",
        )

        affine_trans = loader.add_mutually_exclusive_group(required=False)
        affine_trans.add_argument(
            "--affine_trans",
            dest="affine_trans",
            action="store_true",
            help="Use affine transformations during training",
        )
        affine_trans.add_argument(
            "--no-affine_trans",
            dest="affine_trans",
            action="store_false",
            help="Do not Use affine transformations during training",
        )
        affine_trans.set_defaults(affine_trans=True)

        only_table = loader.add_mutually_exclusive_group(required=False)
        only_table.add_argument(
            "--only_table",
            dest="only_table",
            action="store_true",
            help="Use affine transformations during training",
        )
        only_table.add_argument(
            "--no-only_table",
            dest="only_table",
            action="store_false",
            help="Do not Use affine transformations during training",
        )
        only_table.set_defaults(affine_trans=False)

        loader.add_argument(
            "--t_stdv",
            default=0.02,
            type=float,
            help="std deviation of normal dist. used in translate",
        )
        loader.add_argument(
            "--r_kappa",
            default=30,
            type=float,
            help="concentration of von mises dist. used in rotate",
        )
        loader.add_argument(
            "--sc_stdv",
            default=0.12,
            type=float,
            help="std deviation of log-normal dist. used in scale",
        )
        loader.add_argument(
            "--sh_kappa",
            default=20,
            type=float,
            help="concentration of von mises dist. used in shear",
        )
        loader.add_argument(
            "--trans_prob",
            default=0.8,
            type=float,
            help="probabiliti to perform a transformation",
        )
        loader.add_argument(
            "--do_prior",
            default=False,
            type=bool,
            help="Compute prior distribution over classes",
        )
        loader.add_argument(
            "--load_model",
            default=True,
            type=bool,
            help="Compute prior distribution over classes",
        )
        # ----------------------------------------------------------------------
        # ----- Define NN parameters
        # ----------------------------------------------------------------------
        net = self.parser.add_argument_group("Neural Networks Parameters")
        net.add_argument(
            "--input_channels",
            default=3,
            type=int,
            help="Number of channels of input data",
        )
        net.add_argument(
            "--type_net",
            default="normal",
            type=str,
            choices=["normal", "dense", "residual"],
            help="Type of net. If use dense or residual, modify n_blocks",
        )
        net.add_argument(
            "--n_blocks",
            default=3,
            type=int,
            help="Number of channels for the output",
        )
        net.add_argument(
            "--output_channels",
            default=2,
            type=int,
            help="Number of channels for the output",
        )
        net.add_argument(
            "--heads_att",
            default=8,
            type=int,
            help="Number of channels of input data",
        )
        net.add_argument(
            "--dk",
            default=32,
            type=int,
            help="Number of channels of input data",
        )
        net.add_argument(
            "--dv",
            default=32,
            type=int,
            help="Number of channels of input data",
        )
        net.add_argument(
            "--cnn_ngf", default=12, type=int, help="Number of filters of CNNs"
        )
        n_meg = net.add_mutually_exclusive_group(required=False)

        net.add_argument(
            "--g_loss",
            default="NLL",
            type=str,
            choices=["BCE", "NLL"],
            help="Loss function",
        )

        # ----------------------------------------------------------------------
        # ----- Define Optimizer parameters
        # ----------------------------------------------------------------------
        optim = self.parser.add_argument_group("Optimizer Parameters")
        optim.add_argument(
            "--adam_lr",
            default=0.001,
            type=float,
            help="Initial Lerning rate for ADAM opt",
        )
        optim.add_argument(
            "--adam_beta1",
            default=0.5,
            type=float,
            help="First ADAM exponential decay rate",
        )
        optim.add_argument(
            "--adam_beta2",
            default=0.999,
            type=float,
            help="Secod ADAM exponential decay rate",
        )
        optim.add_argument(
            "--alpha_mae",
            default=0.001,
            type=float,
            help="Alpha to ponderate the loss function on skewing",
        )
        # ----------------------------------------------------------------------
        # ----- Define Train parameters
        # ----------------------------------------------------------------------
        train = self.parser.add_argument_group("Training Parameters")


        skew = train.add_mutually_exclusive_group(required=False)
        skew.add_argument(
            "--do_skew", dest="do_skew", action="store_true", help="Run train stage"
        )
        skew.add_argument(
            "--no-skew",
            dest="do_skew",
            action="store_false",
            help="Do not run train stage",
        )
        skew.set_defaults(do_skew=True)



        debug = train.add_mutually_exclusive_group(required=False)
        debug.add_argument(
            "--debug", dest="debug", action="store_true", help="Run train stage"
        )
        debug.add_argument(
            "--no-debug",
            dest="debug",
            action="store_false",
            help="Do not run train stage",
        )
        debug.set_defaults(debug=False)

        only_blines = train.add_mutually_exclusive_group(required=False)
        only_blines.add_argument(
            "--only_blines", dest="only_blines", action="store_true", help="Run train stage"
        )
        only_blines.add_argument(
            "--not-only_blines",
            dest="only_blines",
            action="store_false",
            help="Do not run train stage",
        )
        only_blines.set_defaults(only_blines=False)

        with_lines = train.add_mutually_exclusive_group(required=False)
        with_lines.add_argument(
            "--with_lines", dest="with_lines", action="store_true", help="Run train stage"
        )
        with_lines.add_argument(
            "--not-with_lines",
            dest="with_lines",
            action="store_false",
            help="Do not run train stage",
        )
        with_lines.set_defaults(with_lines=False)

        with_rc = train.add_mutually_exclusive_group(required=False)
        with_rc.add_argument(
            "--with_rc", dest="with_rc", action="store_true", help="Run train stage"
        )
        with_rc.add_argument(
            "--not-with_rc",
            dest="with_rc",
            action="store_false",
            help="Do not run train stage",
        )
        with_rc.set_defaults(with_rc=False)

        only_cols = train.add_mutually_exclusive_group(required=False)
        only_cols.add_argument(
            "--only_cols", dest="only_cols", action="store_true", help="only_cols and the table shape"
        )
        only_cols.add_argument(
            "--not-only_cols",
            dest="only_cols",
            action="store_false",
            help="Do not run train stage",
        )
        only_cols.set_defaults(only_cols=False)

        train.add_argument(
            "--rc_data",
            default="./data/train/",
            type=str,
            help="""Train rc_data folder. Pkl's""",
        )

        train.add_argument(
            "--cont_train",
            default=False,
            action="store_true",
            help="Continue training using this model",
        )
        train.add_argument(
            "--prev_model",
            default=None,
            type=str,
            help="Use this previously trainned model",
        )
        train.add_argument(
            "--knn",
            default=0,
            type=int,
            help="number of kNN. This option makes the network dynamic",
        )
        train.add_argument(
            "--save_rate",
            default=10,
            type=int,
            help="Save checkpoint each --save_rate epochs",
        )
        train.add_argument(
            "--data_path",
            default="./data/train/",
            type=str,
            help="""Train data folder.""",
        )
        train.add_argument(
            "--fold_paths",
            default="",
            type=str,
            help="""Folder with fold files. Will search with regexp fold*txt. If its empty randomly 4fold CV will be done""",
        )
        train.add_argument(
            "--GL_type",
            default="abs_diff",
            type=str,
            help="""Folder with fold files. Will search with regexp fold*txt. If its empty randomly 4fold CV will be done""",
        )
        train.add_argument(
            "--epochs", default=8000, type=int, help="Number of training epochs"
        )

        train.add_argument(
            "--fix_class_imbalance",
            default=True,
            type=bool,
            help="use weights at loss function to handle class imbalance.",
        )
        train.add_argument(
            "--weight_const",
            default=1.02,
            type=float,
            help="weight constant to fix class imbalance",
        )
        # ----------------------------------------------------------------------
        # ----- Define Test parameters
        # ----------------------------------------------------------------------
        test = self.parser.add_argument_group("Test Parameters")

        te_save = test.add_mutually_exclusive_group(required=False)
        te_save.add_argument(
            "--save_test", dest="save_test", action="store_true", help="Save the result as pickle file"
        )
        te_save.add_argument(
            "--no-save_test",
            dest="save_test",
            action="store_false",
            help="Dont Save the result as pickle file",
        )
        te_save.set_defaults(save_test=False)

        test.add_argument(
            "--do_off",
            default=True,
            type=bool,
            help="Turn DropOut Off during inference",
        )
        test.add_argument(
            "--myfolds",
            default=True,
            type=bool,
            help="Select between splits. True = our folds",
        )
        # ----------------------------------------------------------------------
        # ----- Define PRODUCTION parameters
        # ----------------------------------------------------------------------
        prod = self.parser.add_argument_group("Prod Parameters")
        prod_meg = prod.add_mutually_exclusive_group(required=False)
        prod_meg.add_argument(
            "--do_prod", dest="do_prod", action="store_true", help="Run test stage"
        )
        prod_meg.add_argument(
            "--no-do_prod",
            dest="do_prod",
            action="store_false",
            help="Do not run test stage",
        )
        prod_meg.set_defaults(do_prod=False)
        prod.add_argument(
            "--prod_data",
            default="./data/prod/",
            type=str,
            help="""Prod data folder.""",
        )
        prod.add_argument(
            "--dpi",
            default=300,
            type=int,
            help="""Prod data folder.""",
        )
        # ----------------------------------------------------------------------
        # ----- Define Validation parameters
        # ----------------------------------------------------------------------
        validation = self.parser.add_argument_group("Validation Parameters")
        v_meg = validation.add_mutually_exclusive_group(required=False)
        v_meg.add_argument(
            "--do_val", dest="do_val", action="store_true", help="Run Validation stage"
        )
        v_meg.add_argument(
            "--no-do_val",
            dest="do_val",
            action="store_false",
            help="do not run Validation stage",
        )
        v_meg.set_defaults(do_val=False)

        # ----------------------------------------------------------------------
        # ----- Define Evaluation parameters
        # ----------------------------------------------------------------------
        evaluation = self.parser.add_argument_group("Evaluation Parameters")
        evaluation.add_argument(
            "--target_list",
            default="",
            type=str,
            help="List of ground-truth PAGE-XML files",
        )
        evaluation.add_argument(
            "--hyp_list", default="", type=str, help="List of hypotesis PAGE-XMLfiles"
        )

    def _convert_file_to_args(self, arg_line):
        return arg_line.split(" ")

    def _str_to_bool(self, data):
        """
        Nice way to handle bool flags:
        from: https://stackoverflow.com/a/43357954
        """
        if data.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif data.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    def _check_out_dir(self, pointer):
        """ Checks if the dir is wirtable"""
        if os.path.isdir(pointer):
            # --- check if is writeable
            if os.access(pointer, os.W_OK):
                if not (os.path.isdir(pointer + "/checkpoints")):
                    os.makedirs(pointer + "/checkpoints")
                    self.logger.debug(
                        "Creating checkpoints dir: {}".format(pointer + "/checkpoints")
                    )
                return pointer
            else:
                raise argparse.ArgumentTypeError(
                    "{} folder is not writeable.".format(pointer)
                )
        else:
            try:
                os.makedirs(pointer)
                self.logger.debug("Creating output dir: {}".format(pointer))
                os.makedirs(pointer + "/checkpoints")
                self.logger.debug(
                    "Creating checkpoints dir: {}".format(pointer + "/checkpoints")
                )
                return pointer
            except OSError as e:
                raise argparse.ArgumentTypeError(
                    "{} folder does not exist and cannot be created\n{}".format(e)
                )

    def _check_in_dir(self, pointer):
        """check if path exists and is readable"""
        if os.path.isdir(pointer):
            if os.access(pointer, os.R_OK):
                return pointer
            else:
                raise argparse.ArgumentTypeError(
                    "{} folder is not readable.".format(pointer)
                )
        else:
            raise argparse.ArgumentTypeError(
                "{} folder does not exists".format(pointer)
            )

    def _check_to_int_array(self, data):
        """check is size is 256 multiple"""
        data = int(data)
        if data > 0 and data % 256 == 0:
            return data
        else:
            raise argparse.ArgumentTypeError(
                "Image size must be multiple of 256: {} is not".format(data)
            )
    def create_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def parse(self):
        """Perform arguments parsing"""
        # --- Parse initialization + command line arguments
        # --- Arguments priority stack:
        # ---    1) command line arguments
        # ---    2) config file arguments
        # ---    3) default arguments
        self.opts, unkwn = self.parser.parse_known_args()
        if unkwn:
            msg = "unrecognized command line arguments: {}\n".format(unkwn)
            print(msg)
            self.parser.error(msg)

        # --- Parse config file if defined
        if self.opts.config != None:
            self.logger.info("Reading configuration from {}".format(self.opts.config))
            self.opts, unkwn_conf = self.parser.parse_known_args(
                ["@" + self.opts.config], namespace=self.opts
            )
            if unkwn_conf:
                msg = "unrecognized  arguments in config file: {}\n".format(unkwn_conf)
                self.parser.error(msg)
            self.opts = self.parser.parse_args(namespace=self.opts)
        # --- Preprocess some input variables
        # --- enable/disable
        self.opts.use_gpu = self.opts.gpu != -1
        layers = self.opts.layers
        self.opts.layers = [int(x) for x in layers.split(",")]
        layers_MLP = self.opts.layers_MLP
        self.opts.layers_MLP = [int(x) for x in layers_MLP.split(",")]
        # --- make sure to don't use pinned memory when CPU only, DataLoader class
        # --- will copy tensors into GPU by default if pinned memory is True.
        if not self.opts.use_gpu:
            self.opts.pin_memory = False
        # --- set logging data
        self.opts.log_level_id = getattr(logging, self.opts.log_level.upper())
        self.opts.log_file = self.opts.work_dir + "/" + self.opts.exp_name + ".log"
        # --- add merde regions to color dic, so parent and merged will share the same color

        # --- TODO: Move this create dir to check inputs function
        self._check_out_dir(self.opts.work_dir)
        self.opts.checkpoints = os.path.join(self.opts.work_dir, "checkpoints/")
        if self.opts.debug:
            self.create_dir(os.path.join(self.opts.work_dir, "debug/"))
            self.create_dir(os.path.join(self.opts.work_dir, "debug/test"))
            self.create_dir(os.path.join(self.opts.work_dir, "debug/train"))
            self.create_dir(os.path.join(self.opts.work_dir, "debug/dev"))
        # if self.opts.do_class:
        #    self.opts.line_color = 1
        # --- define network output channels based on inputs
        return self.opts

    def __str__(self):
        """pretty print handle"""
        data = "------------ Options -------------"
        try:
            for k, v in sorted(vars(self.opts).items()):
                data = data + "\n" + "{0:15}\t{1}".format(k, v)
        except:
            data = data + "\nNo arguments parsed yet..."

        data = data + "\n---------- End  Options ----------\n"
        return data

    def __repr__(self):
        return self.__str__()
