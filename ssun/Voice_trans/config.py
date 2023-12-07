import argparse
import os

def dataset_info(network_name):
    dataset_info = dict()
    if network_name == 'voice_trans':
        dataset_info['dataset_path_train'] = '/storage/mskim/English_voice/train/'
        dataset_info['dataset_path_test'] = '/storage/mskim/English_voice/test/'
        dataset_info['dataset_path'] = '/storage/mskim/English_voice/'
        dataset_info['batch_size'] = 2
        dataset_info['lambda_r'] = 1
        dataset_info['lambda_c'] = 1
        dataset_info['lambda_p'] = 1
        dataset_info['n_bins'] = 256

    else:
        ValueError('There is no dataset named {}'.format(network_name))
    return dataset_info


class Config:
    map_path = './confusion_map/map_data/'

    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.parser.add_argument('--network_name', type=str, default='voice_trans')
        self.parser.add_argument('--weight_name', type=str, default='voice_trans')
        self.parser.add_argument('--dataset_name', type=str, default='dacon')
        self.parser.add_argument('--tensor_name', type=str, default='dis_star')
        self.parser.add_argument('--checkpoint_name', type=str, default='dis_star')
        self.parser.add_argument('--continue_train', type=bool, default=False)
        self.parser.add_argument('--epochs', type=int, default=100)
        self.parser.add_argument('--checkpoint_load_num', type=int, default=500)

        parser, _ = self.parser.parse_known_args()
        self.dataset_info = dataset_info(network_name=parser.network_name)

        self.parser.add_argument('--batch_size', type=int, default=self.dataset_info['batch_size'])
        self.parser.add_argument('--dataset_path_train', type=str, default=self.dataset_info['dataset_path_train'])
        self.parser.add_argument('--dataset_path_test', type=str, default=self.dataset_info['dataset_path_test'])
        self.parser.add_argument('--dataset_path', type=str, default=self.dataset_info['dataset_path'])
        self.parser.add_argument('--lambda_r', type=float, default=self.dataset_info['lambda_r'])
        self.parser.add_argument('--lambda_c', type=float, default=self.dataset_info['lambda_c'])
        self.parser.add_argument('--lambda_p', type=float, default=self.dataset_info['lambda_p'])
        self.parser.add_argument('--n_bins', type=float, default=self.dataset_info['n_bins'])

        self.parser.add_argument('--scheduler_name', type=str, default='cosine', help='[stepLR | cycliclr | cosine]')
        self.parser.add_argument('--lr', type=float, default=1e-6)
        self.parser.add_argument('--optimizer_name', type=str, default='Adam', help='[Adam | RMSprop]')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--momentum', type=float, default=0.9, help='monument for rmsprop optimizer')
        self.parser.add_argument('--weight_decay', type=float, default=5e-5, help='weight decay')

        self.parser.add_argument('--save_path', type=str, default='./checkpoints/pre_test_{}_{}'.format(parser.dataset_name, parser.network_name), help='path to store model')
        self.parser.add_argument('--train_test_save_path', type=str, default='./train_test/' + parser.network_name, help='')
        self.parser.add_argument('--use_cuda', type=bool, default=True, help='use cuda')
        self.parser.add_argument('--gpu_id', type=str, default='0', help='gpu id used to train')
        self.parser.add_argument('--phase', type=str, default='train')
        self.parser.add_argument('--num_workers', type=int, default=0)
        self.parser.add_argument('--samplier', type=int, default=1)
        self.parser.add_argument('--debugging', type=bool, default=False)
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
