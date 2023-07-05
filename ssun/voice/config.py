import argparse
import os

def dataset_info(dataset_name):
    dataset_info = dict()
    if  dataset_name == 'speechsplit':
        dataset_info['dataset_path_train'] = '/storage/mskim/English_voice/train/'
        dataset_info['dataset_path_test'] = '/storage/mskim/English_voice/test/'
        dataset_info['dataset_path_make'] = '/storage/mskim/English_voice/make_dataset/'

        dataset_info['freq'] = 8
        dataset_info['dim_neck'] = 8
        dataset_info['freq_2'] = 8
        dataset_info['dim_neck_2'] = 1
        dataset_info['freq_3'] = 8
        dataset_info['dim_neck_3'] = 32
        dataset_info['out_channels'] = 30
        dataset_info['layers'] = 24
        dataset_info['stacks'] = 4
        dataset_info['residual_channels'] = 512
        dataset_info['gate_channels'] = 512
        dataset_info['skip_out_channels'] = 256
        dataset_info['cin_channels'] = 8
        dataset_info['gin_channels'] = 1

        dataset_info['weight_normalization'] = True
        dataset_info['n_speakers'] = 1
        dataset_info['dropout'] = 0.05
        dataset_info['kernel_size'] = 3
        dataset_info['upsample_conditional_features'] = True
        dataset_info['upsample_scales'] = [4, 4, 4, 4]
        dataset_info['freq_axis_kernel_size'] = 3
        dataset_info['legacy'] = True

        dataset_info['dim_enc'] = 512
        dataset_info['dim_enc_2'] = 128
        dataset_info['dim_enc_3'] = 256

        dataset_info['dim_freq'] = 80
        dataset_info['dim_spk_emb'] = 82
        dataset_info['dim_f0'] = 257
        dataset_info['dim_dec'] = 512
        dataset_info['len_raw'] = 128
        dataset_info['chs_grp'] = 16

        dataset_info['min_len_seg'] = 19
        dataset_info['max_len_seg'] = 32
        dataset_info['min_len_seq'] = 64
        dataset_info['max_len_seq'] = 128
        dataset_info['max_len_pad'] = 192


    else:
        ValueError('There is no dataset named {}'.format(dataset_name))
    return dataset_info


class Config:
    map_path = './confusion_map/map_data/'

    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.parser.add_argument('--network_name', type=str, default='capsnet') # [ resnet | densenet | efficientnet | capsnet ]
        self.parser.add_argument('--weight_name', type=str, default='capsnet') # [ resnet | densenet | efficientnet | capsnet ]
        self.parser.add_argument('--dataset_name', type=str, default='multi')
        self.parser.add_argument('--continue_train', type=bool, default=False)
        self.parser.add_argument('--epochs', type=int, default=20)
        #
        temp_parser, _ = self.parser.parse_known_args()
        self.dataset_info = dataset_info(dataset_name=temp_parser.dataset_name)
        #
        self.parser.add_argument('--batch_size', type=int, default=self.dataset_info['batch_size'])
        self.parser.add_argument('--dataset_path_train', type=str, default=self.dataset_info['dataset_path_train'])
        self.parser.add_argument('--dataset_path_test', type=str, default=self.dataset_info['dataset_path_test'])
        self.parser.add_argument('--loss_name', type=str, default=self.dataset_info['loss_name'])

        self.parser.add_argument('--data_height', type=int, default=self.dataset_info['data_height'])
        self.parser.add_argument('--data_width', type=int, default=self.dataset_info['data_width'])
        self.parser.add_argument('--data_depth', type=int, default=self.dataset_info['data_depth'])

        self.parser.add_argument('--in_dim', type=int, default=self.dataset_info['in_dim'])
        self.parser.add_argument('--out_dim', type=int, default=self.dataset_info['out_dim'])
        self.parser.add_argument('--out_channels', type=int, default=self.dataset_info['out_channels'])
        self.parser.add_argument('--num_routing', type=int, default=self.dataset_info['num_routing'])
        self.parser.add_argument('--threshold', type=float, default=self.dataset_info['threshold'])
        #####
        self.parser.add_argument('--scheduler_name', type=str, default='cosine', help='[stepLR | cycliclr | cosine]')
        self.parser.add_argument('--lr', type=float, default=1e-4)
        self.parser.add_argument('--optimizer_name', type=str, default='Adam', help='[Adam | RMSprop]')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--momentum', type=float, default=0.9, help='monument for rmsprop optimizer')
        self.parser.add_argument('--weight_decay', type=float, default=5e-5, help='weight decay')
        #####
        self.parser.add_argument('--save_path', type=str, default='./checkpoints/pre_test_{}_{}'.format(temp_parser.dataset_name, temp_parser.network_name), help='path to store model')
        self.parser.add_argument('--train_test_save_path', type=str, default='./train_test/' + temp_parser.network_name, help='')
        self.parser.add_argument('--use_cuda', type=bool, default=True, help='use cuda')
        self.parser.add_argument('--gpu_id', type=str, default='1', help='gpu id used to train')
        self.parser.add_argument('--phase', type=str, default='train')
        self.parser.add_argument('--freq_show_loss', type=int, default=100)
        self.parser.add_argument('--freq_show_image', type=int, default=200)
        self.parser.add_argument('--freq_save_net', type=int, default=50)
        self.parser.add_argument('--num_test_iter', type=int, default=5)

        self.opt, _ = self.parser.parse_known_args()

    def print_options(self):
        """Print and save options
                It will print both current options and default values(if different).
                It will save options into a text file / [checkpoints_dir] / opt.txt
                """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(self.opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(self.opt.save_path)
        mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(self.opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')


def mkdirs(paths):
    """create empty directories if they don't exist
    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
