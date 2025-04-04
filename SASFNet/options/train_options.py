from .base_options import BaseOptions


class TrainOptions(BaseOptions):
	def initialize(self):
		BaseOptions.initialize(self)
		self.parser.add_argument('--display_freq', type=int, default=500, help='frequency of showing training bi_lstm_results on screen')
		self.parser.add_argument('--print_freq', type=int, default=500, help='frequency of showing training bi_lstm_results on console')
		self.parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest bi_lstm_results')
		self.parser.add_argument('--save_epoch_freq', type=int, default=100, help='frequency of saving checkpoints at the end of epochs')
		########################## 常改 #############
		self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
		self.parser.add_argument('--epoch_count', type=int, default=0, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
		self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
		self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
		self.parser.add_argument('--niter', type=int, default=1400, help='# of iter at starting learning rate')
		self.parser.add_argument('--niter_decay', type=int, default=2000, help='# of iter to linearly decay learning rate to zero')
		self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
		self.parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')#0.0001
		self.parser.add_argument('--lambda_A', type=float, default=100.0, help='weight for cycle loss (A -> B -> A)')
		self.parser.add_argument('--lambda_B', type=float, default=50.0, help='weight for cycle loss (B -> A -> B)')
		self.parser.add_argument('--identity', type=float, default=0.0, help='use identity mapping. Setting identity other than 1 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set optidentity = 0.1')
		self.parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
		self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training bi_lstm_results to [opt.checkpoints_dir]/[opt.name]/web/')
		self.isTrain = True
