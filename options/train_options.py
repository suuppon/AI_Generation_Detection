from .base_options import BaseOptions

 
class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--earlystop_epoch', type=int, default=15)
        parser.add_argument('--data_aug', action='store_true', help='if specified, perform additional data augmentation (photometric, blurring, jpegging)')
        parser.add_argument('--optim', type=str, default='adam', help='optim to use [sgd, adam]')
        parser.add_argument('--new_optim', action='store_true', help='new optimizer instead of loading the optim state')
        parser.add_argument('--loss_freq', type=int, default=400, help='frequency of showing loss on tensorboard')
        parser.add_argument('--save_latest_freq', type=int, default=2000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=20, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--last_epoch', type=int, default=-1, help='starting epoch count for scheduler intialization')
        parser.add_argument('--train_split', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--val_split', type=str, default='val', help='train, val, test, etc')
        parser.add_argument('--epochs', type=int, default=1000, help='# of iter at starting learning rate')
        parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        # parser.add_argument('--model_path')
        # parser.add_argument('--no_resize', action='store_true')
        # parser.add_argument('--no_crop', action='store_true')
        
        parser.add_argument('--weight_decay', type=float, default=0.0, help='L2 weight decay')
        parser.add_argument('--sched', type=str, default='cosine', choices=['cosine','step','none'],
                            help='lr scheduler type')
        parser.add_argument('--warmup_steps', type=int, default=0, help='warmup steps for scheduler')
        parser.add_argument('--save_best_metric', type=str, default='val_acc',
                            help='metric name to track best checkpoint (e.g., val_acc, val_auc, val_loss)')
        self.isTrain = True
        return parser
