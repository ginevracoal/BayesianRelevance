import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

from tqdm import tqdm
import random 
from torch.utils.data import Subset, DataLoader
import bayesian_torch.bayesian_torch.models.bayesian.resnet_variational as resnet
from attacks.run_attacks import run_attack, save_attack, load_attack
from utils.lrp import *

model_names = sorted(
	name for name in resnet.__dict__
	if name.islower() and not name.startswith("__")
	and name.startswith("resnet") and callable(resnet.__dict__[name]))

print(model_names)
len_trainset = 50000
len_testset = 10000

parser = argparse.ArgumentParser(description='CIFAR10')
parser.add_argument('--arch',
					'-a',
					metavar='ARCH',
					default='resnet20',
					choices=model_names,
					help='model architecture: ' + ' | '.join(model_names) +
					' (default: resnet20)')
parser.add_argument('-j',
					'--workers',
					default=8,
					type=int,
					metavar='N',
					help='number of data loading workers (default: 8)')
parser.add_argument('--epochs',
					default=200,
					type=int,
					metavar='N',
					help='number of total epochs to run')
parser.add_argument('--start-epoch',
					default=0,
					type=int,
					metavar='N',
					help='manual epoch number (useful on restarts)')
parser.add_argument('--lr',
					'--learning-rate',
					default=0.001,
					type=float,
					metavar='LR',
					help='initial learning rate')
parser.add_argument('--momentum',
					default=0.9,
					type=float,
					metavar='M',
					help='momentum')
parser.add_argument('--weight-decay',
					'--wd',
					default=1e-4,
					type=float,
					metavar='W',
					help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq',
					'-p',
					default=50,
					type=int,
					metavar='N',
					help='print frequency (default: 20)')
parser.add_argument('--resume',
					default='',
					type=str,
					metavar='PATH',
					help='path to latest checkpoint (default: none)')
parser.add_argument('-e',
					'--evaluate',
					dest='evaluate',
					action='store_true',
					help='evaluate model on validation set')
parser.add_argument('--pretrained',
					dest='pretrained',
					action='store_true',
					help='use pre-trained model')
parser.add_argument('--half',
					dest='half',
					action='store_true',
					help='use half-precision(16-bit) ')
parser.add_argument('--save-dir',
					dest='save_dir',
					help='The directory used to save the trained models',
					default='../experiments/fullBNN/cifar_resnet/',
					type=str)
parser.add_argument(
	'--save-every',
	dest='save_every',
	help='Saves checkpoints at every specified number of epochs',
	type=int,
	default=10)
parser.add_argument('--num_mc',
					type=int,
					default=5,
					metavar='N',
					help='number of Monte Carlo runs during training')
parser.add_argument(
	'--tensorboard',
	type=bool,
	default=False,
	metavar='N',
	help='use tensorboard for logging and visualization of training progress')
parser.add_argument(
	'--log_dir',
	type=str,
	default='./bayesian_torch/logs/cifar/bayesian',
	metavar='N',
	help='use tensorboard for logging and visualization of training progress')
parser.add_argument('--mode', type=str, default='test', help='train | test')
parser.add_argument(
	'--n_samples',
	type=int,
	default=100,
	metavar='N',
	help='number of Monte Carlo samples to be drawn during inference')
parser.add_argument(
	'--attack_method',
	type=str, 
	default='fgsm',
	help='fgsm, pgd')
parser.add_argument(
	'--test_inputs',
	type=int, 
	default=500)


best_prec1 = 0
attack_hyperparams={'epsilon':0.2}


def MOPED_layer(layer, det_layer, delta):
	"""
	Set the priors and initialize surrogate posteriors of Bayesian NN with Empirical Bayes
	MOPED (Model Priors with Empirical Bayes using Deterministic DNN)
	Reference:
	[1] Ranganath Krishnan, Mahesh Subedar, Omesh Tickoo.
		Specifying Weight Priors in Bayesian Deep Neural Networks with Empirical Bayes. AAAI 2020.
	"""

	if (str(layer) == 'Conv2dReparameterization()'):
		#set the priors
		print(str(layer))
		layer.prior_weight_mu = det_layer.weight.data
		if layer.prior_bias_mu is not None:
			layer.prior_bias_mu = det_layer.bias.data

		#initialize surrogate posteriors
		layer.mu_kernel.data = det_layer.weight.data
		layer.rho_kernel.data = get_rho(det_layer.weight.data, delta)
		if layer.mu_bias is not None:
			layer.mu_bias.data = det_layer.bias.data
			layer.rho_bias.data = get_rho(det_layer.bias.data, delta)

	elif (isinstance(layer, nn.Conv2d)):
		print(str(layer))
		layer.weight.data = det_layer.weight.data
		if layer.bias is not None:
			layer.bias.data = det_layer.bias.data

	elif (str(layer) == 'LinearReparameterization()'):
		print(str(layer))
		layer.prior_weight_mu = det_layer.weight.data
		if layer.prior_bias_mu is not None:
			layer.prior_bias_mu = det_layer.bias.data

		#initialize the surrogate posteriors
		layer.mu_weight.data = det_layer.weight.data
		layer.rho_weight.data = get_rho(det_layer.weight.data, delta)
		if layer.mu_bias is not None:
			layer.mu_bias.data = det_layer.bias.data
			layer.rho_bias.data = get_rho(det_layer.bias.data, delta)

	elif str(layer).startswith('Batch'):
		#initialize parameters
		print(str(layer))
		layer.weight.data = det_layer.weight.data
		if layer.bias is not None:
			layer.bias.data = det_layer.bias.data
		layer.running_mean.data = det_layer.running_mean.data
		layer.running_var.data = det_layer.running_var.data
		layer.num_batches_tracked.data = det_layer.num_batches_tracked.data


def main():
	global args, best_prec1
	args = parser.parse_args()

	# Check the save_dir exists or not
	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)

	model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
	if torch.cuda.is_available():
		model.cuda()
	else:
		model.cpu()

	# optionally resume from a checkpoint
	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume)
			args.start_epoch = checkpoint['epoch']
			best_prec1 = checkpoint['best_prec1']
			model.load_state_dict(checkpoint['state_dict'])
			print("=> loaded checkpoint '{}' (epoch {})".format(
				args.evaluate, checkpoint['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))

	cudnn.benchmark = True

	tb_writer = None
	# if args.tensorboard:
	#     logger_dir = os.path.join(args.log_dir, 'tb_logger')
	#     if not os.path.exists(logger_dir):
	#         os.makedirs(logger_dir)
	#     tb_writer = SummaryWriter(logger_dir)

	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									 std=[0.229, 0.224, 0.225])

	batch_size = 128 if args.mode=='train' else 100

	train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(
		root='../experiments/bayesian_torch/data',
		train=True,
		transform=transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.RandomCrop(32, 4),
			transforms.ToTensor(),
			normalize,
		]),
		download=True),
											   batch_size=batch_size,
											   shuffle=True,
											   num_workers=args.workers,
											   pin_memory=True)

	val_loader = torch.utils.data.DataLoader(datasets.CIFAR10(
		root='../experiments/bayesian_torch/data',
		train=False,
		transform=transforms.Compose([
			transforms.ToTensor(),
			normalize,
		])),
											 batch_size=batch_size,
											 shuffle=False,
											 num_workers=args.workers,
											 pin_memory=True)

	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)

	if torch.cuda.is_available():
		criterion = nn.CrossEntropyLoss().cuda()
	else:
		criterion = nn.CrossEntropyLoss().cpu()

	if args.half:
		model.half()
		criterion.half()

	if args.arch in ['resnet110']:
		for param_group in optimizer.param_groups:
			param_group['lr'] = args.lr * 0.1

	if args.evaluate:
		validate(val_loader, model, criterion)
		return

	if args.mode == 'train':

		for epoch in range(args.start_epoch, args.epochs):

			lr = args.lr
			if (epoch >= 80 and epoch < 120):
				lr = 0.1 * args.lr
			elif (epoch >= 120 and epoch < 160):
				lr = 0.01 * args.lr
			elif (epoch >= 160 and epoch < 180):
				lr = 0.001 * args.lr
			elif (epoch >= 180):
				lr = 0.0005 * args.lr

			optimizer = torch.optim.Adam(model.parameters(), lr)

			# train for one epoch
			print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
			train(args, train_loader, model, criterion, optimizer, epoch,
				  tb_writer)

			prec1 = validate(args, val_loader, model, criterion, epoch,
							 tb_writer)

			is_best = prec1 > best_prec1
			best_prec1 = max(prec1, best_prec1)

			if is_best:
				save_checkpoint(
					{
						'epoch': epoch + 1,
						'state_dict': model.state_dict(),
						'best_prec1': best_prec1,
					},
					is_best,
					filename=os.path.join(
						args.save_dir,
						'bayesian_{}_cifar.pth'.format(args.arch)))

	elif args.mode == 'test':

		checkpoint_file = args.save_dir + '/bayesian_{}_cifar.pth'.format(
			args.arch)
		if torch.cuda.is_available():
			checkpoint = torch.load(checkpoint_file)
			device="cuda"
		else:
			checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
			device="cpu"

		# print(model.state_dict().keys())
		# print(checkpoint['state_dict'].keys())
		# exit()

		state_dict = checkpoint['state_dict']
		model.load_state_dict(state_dict)
		evaluate(args, model, val_loader, n_samples=args.n_samples)

		# model = convert_resnet(model).to(device)

		# Adversarial attacks

		method=args.attack_method
		test_inputs = args.test_inputs
		n_samples = args.n_samples
		print(f"\nn_samples = {n_samples}")

		dataset = Subset(val_loader.dataset, range(test_inputs))
		images, labels = ([],[])
		for image, label in dataset: 
			images.append(image)
			labels.append(label)
		images = torch.stack(images)

		bay_attack = attack(model, dataset, n_samples=n_samples, method=method, hyperparams=attack_hyperparams)
		save_attack(inputs=images, attacks=bay_attack, method=method, model_savedir=args.save_dir, n_samples=n_samples)
		# bay_attack = load_attack(method=method, model_savedir=args.save_dir, n_samples=n_samples)

		# print(images.min(), images.max(), bay_attack.min(), bay_attack.max())

		# evaluate(args, model, DataLoader(dataset=list(zip(images, labels))), n_samples=n_samples)
		# evaluate(args, model, DataLoader(dataset=list(zip(bay_attack, labels))), n_samples=n_samples)

		# LRP

		for rule in ['epsilon','gamma','alpha1beta0']:
			learnable_layers_idxs=[38] #[0,2,14,26,38]

			for layer_idx in learnable_layers_idxs:

				print(f"\nlayer_idx = {layer_idx}")

				savedir = get_lrp_savedir(model_savedir=args.save_dir, attack_method=method, 
											layer_idx=layer_idx, rule=rule)

				bay_lrp = compute_lrp(images, model, rule=rule, n_samples=n_samples, device=device)
				bay_attack_lrp = compute_lrp(bay_attack, model, 
										device=device, rule=rule, n_samples=n_samples)

				save_to_pickle(bay_lrp, path=savedir, filename="bay_lrp_samp="+str(n_samples))
				save_to_pickle(bay_attack_lrp, path=savedir, filename="bay_attack_lrp_samp="+str(n_samples))
				
				# bay_lrp = load_from_pickle(path=savedir, filename="bay_lrp_samp="+str(n_samples))
				# bay_attack_lrp = load_from_pickle(path=savedir, filename="bay_attack_lrp_samp="+str(n_samples))

			set_seed(0)
			idxs = np.random.choice(len(images), 10, replace=False)
			original_images_plot = torch.stack([images[i].squeeze() for i in idxs])
			adversarial_images_plot = torch.stack([bay_attack[i].squeeze() for i in idxs])
			bay_lrp_heatmaps_plot = torch.stack([bay_lrp[i].squeeze() for i in idxs])
			bay_attack_lrp_heatmaps_plot = torch.stack([bay_attack_lrp[i].squeeze() for i in idxs])
			plot_lrp_grid(original_images=original_images_plot.detach().cpu(), 
						  adversarial_images=adversarial_images_plot.detach().cpu(),
						  bay_lrp_heatmaps=bay_lrp_heatmaps_plot.detach().cpu(),
						  bay_attack_lrp_heatmaps=bay_attack_lrp_heatmaps_plot.detach().cpu(), 
						  filename="lrp_samp="+str(n_samples), savedir=savedir)

def convert_resnet(module, modules=None):
	import torch
	from lrp.sequential import Sequential 
	from lrp.linear import Linear
	from lrp.conv import Conv2d

	conversion_table = { 
		'Linear': Linear,
		'Conv2d': Conv2d
	}

	# First time
	if modules is None: 
		modules = []
		for m in module.children():
			convert_resnet(m, modules=modules)

			# Vgg model has a flatten, which is not represented as a module
			# so this loop doesn't pick it up.
			# This is a hack to make things work.
			if isinstance(m, torch.nn.AdaptiveAvgPool2d): 
				modules.append(torch.nn.Flatten())

		sequential = Sequential(*modules)
		return sequential

	# Recursion
	if isinstance(module, torch.nn.Sequential): 
		for m in module.children():
			convert_vgg(m, modules=modules)

	elif isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
		class_name = module.__class__.__name__
		lrp_module = conversion_table[class_name].from_torch(module)
		modules.append(lrp_module)
	# maxpool is handled with gradient for the moment

	elif isinstance(module, torch.nn.ReLU): 
		# avoid inplace operations. They might ruin PatternNet pattern
		# computations
		modules.append(torch.nn.ReLU())
	else:
		modules.append(module)



def train(args,
		  train_loader,
		  model,
		  criterion,
		  optimizer,
		  epoch,
		  tb_writer=None):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()

	# switch to train mode
	model.train()

	end = time.time()
	for i, (input, target) in enumerate(train_loader):

		# measure data loading time
		data_time.update(time.time() - end)

		if torch.cuda.is_available():
			target = target.cuda()
			input_var = input.cuda()
			target_var = target
		else:
			target = target.cpu()
			input_var = input.cpu()
			target_var = target

		if args.half:
			input_var = input_var.half()

		# compute output
		output_ = []
		kl_ = []
		for mc_run in range(args.num_mc):
			output, kl = model(input_var)
			output_.append(output)
			kl_.append(kl)
		output = torch.mean(torch.stack(output_), dim=0)
		kl = torch.mean(torch.stack(kl_), dim=0)
		cross_entropy_loss = criterion(output, target_var)
		scaled_kl = kl / len_trainset
		#ELBO loss
		loss = cross_entropy_loss + scaled_kl
		'''
		#another way of computing gradients with multiple MC samples
		cross_entropy_loss = 0
		scaled_kl = 0
		for mc_run in range(args.num_mc):
			output, kl = model(input_var)
			cross_entropy_loss += criterion(output, target_var)
			scaled_kl += (kl/len_trainset)
		cross_entropy_loss = cross_entropy_loss/args.num_mc
		scaled_kl = scaled_kl/args.num_mc
		loss = cross_entropy_loss + scaled_kl
		#end
		'''

		# compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		output = output.float()
		loss = loss.float()
		# measure accuracy and record loss
		prec1 = accuracy(output.data, target)[0]
		losses.update(loss.item(), input.size(0))
		top1.update(prec1.item(), input.size(0))

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.print_freq == 0:
			print('Epoch: [{0}][{1}/{2}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
					  epoch,
					  i,
					  len(train_loader),
					  batch_time=batch_time,
					  data_time=data_time,
					  loss=losses,
					  top1=top1))

		if tb_writer is not None:
			tb_writer.add_scalar('train/cross_entropy_loss',
								 cross_entropy_loss.item(), epoch)
			tb_writer.add_scalar('train/kl_div', scaled_kl.item(), epoch)
			tb_writer.add_scalar('train/elbo_loss', loss.item(), epoch)
			tb_writer.add_scalar('train/accuracy', prec1.item(), epoch)
			tb_writer.flush()

def plot_lrp_grid(original_images, adversarial_images, bay_lrp_heatmaps, bay_attack_lrp_heatmaps, filename, savedir):

	import matplotlib.pyplot as plt

	fig, axes = plt.subplots(4, len(original_images), figsize = (12,4))

	for i in range(0, len(original_images)):

		original_image = original_images[i].permute(1,2,0) if len(original_images[i].shape) > 2 else original_images[i]
		adversarial_image = adversarial_images[i].permute(1,2,0) if len(adversarial_images[i].shape) > 2 else adversarial_images[i]
		bay_lrp_heatmap = bay_lrp_heatmaps[i].permute(1,2,0) if len(bay_lrp_heatmaps[i].shape) > 2 else bay_lrp_heatmaps[i]
		bay_attack_lrp_heatmap = bay_attack_lrp_heatmaps[i].permute(1,2,0) if len(bay_attack_lrp_heatmaps[i].shape) > 2 else bay_attack_lrp_heatmaps[i]

		axes[0, i].imshow(torch.clamp(original_image, 0., 1.))
		axes[1, i].imshow(torch.clamp(bay_lrp_heatmap, 0., 1.))
		axes[2, i].imshow(torch.clamp(adversarial_image, 0., 1.))
		axes[3, i].imshow(torch.clamp(bay_attack_lrp_heatmap, 0., 1.))
		
	os.makedirs(os.path.dirname(savedir+"/"), exist_ok=True)
	plt.savefig(os.path.join(savedir, filename+".png"))

def validate(args, val_loader, model, criterion, epoch, tb_writer=None):
	batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()

	# switch to evaluate mode
	model.eval()

	end = time.time()
	with torch.no_grad():
		for i, (input, target) in enumerate(val_loader):
			if torch.cuda.is_available():
				target = target.cuda()
				input_var = input.cuda()
				target_var = target.cuda()
			else:
				target = target.cpu()
				input_var = input.cpu()
				target_var = target.cpu()

			if args.half:
				input_var = input_var.half()

			# compute output
			output_ = []
			kl_ = []
			for mc_run in range(args.num_mc):
				output, kl = model(input_var)
				output_.append(output)
				kl_.append(kl)
			output = torch.mean(torch.stack(output_), dim=0)
			kl = torch.mean(torch.stack(kl_), dim=0)
			cross_entropy_loss = criterion(output, target_var)
			scaled_kl = kl / len_trainset
			#ELBO loss
			loss = cross_entropy_loss + scaled_kl

			output = output.float()
			loss = loss.float()

			# measure accuracy and record loss
			prec1 = accuracy(output.data, target)[0]
			losses.update(loss.item(), input.size(0))
			top1.update(prec1.item(), input.size(0))

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % args.print_freq == 0:
				print('Test: [{0}/{1}]\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
						  i,
						  len(val_loader),
						  batch_time=batch_time,
						  loss=losses,
						  top1=top1))

			if tb_writer is not None:
				tb_writer.add_scalar('val/cross_entropy_loss',
									 cross_entropy_loss.item(), epoch)
				tb_writer.add_scalar('val/kl_div', scaled_kl.item(), epoch)
				tb_writer.add_scalar('val/elbo_loss', loss.item(), epoch)
				tb_writer.add_scalar('val/accuracy', prec1.item(), epoch)
				tb_writer.flush()

	print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

	return top1.avg


def evaluate(args, model, val_loader, n_samples):

	pred_probs_mc = []
	test_loss = 0
	correct = 0
	output_list = []
	labels_list = []

	model.eval()
	with torch.no_grad():
		begin = time.time()
		for data, target in val_loader:

			if torch.cuda.is_available():
				data, target = data.cuda(), target.cuda()
			else:
				data, target = data.cpu(), target.cpu()
			output_mc = []
			for mc_run in range(n_samples):
				random.seed(mc_run)
				output, _ = model.forward(data)
				output_mc.append(output)

			output_ = torch.stack(output_mc)
			output_list.append(output_)
			labels_list.append(target)

		end = time.time()

		print("inference throughput: ", len(val_loader.dataset) / (end - begin),
			  " images/s")

		output = torch.stack(output_list)
		output = output.permute(1, 0, 2, 3)
		output = output.contiguous().view(n_samples, len(val_loader.dataset), -1)
		output = torch.nn.functional.softmax(output, dim=2)
		labels = torch.cat(labels_list)
		pred_mean = output.mean(dim=0)
		Y_pred = torch.argmax(pred_mean, axis=1)
		print('Test accuracy:',
			  (Y_pred.data.cpu().numpy() == labels.data.cpu().numpy()).mean() *
			  100)
		np.save('../experiments/bayesian_torch/probs_cifar_mc.npy', output.data.cpu().numpy())
		np.save('../experiments/bayesian_torch/cifar_test_labels_mc.npy', labels.data.cpu().numpy())

def attack(model, dataset, n_samples, method, hyperparams):

	model.eval()
	adversarial_attacks = []

	for data, target in tqdm(dataset):
		data = data.unsqueeze(0)
		target = torch.tensor(target).unsqueeze(0)

		if torch.cuda.is_available():
			data, target = data.cuda(), target.cuda()
			device = 'cuda'
		else:
			data, target = data.cpu(), target.cpu()
			device = 'cpu'

		samples_attacks=[]

		for idx in list(range(n_samples)):
			random.seed(idx)
			perturbed_image = run_attack(net=model, image=data, label=target, method=method, 
										 device=device, hyperparams=hyperparams).squeeze()
			# print(perturbed_image[0,0,:5])
			perturbed_image = torch.clamp(perturbed_image, 0., 1.)
			samples_attacks.append(perturbed_image.unsqueeze(0))
		# exit()

		adversarial_attack = torch.stack(samples_attacks).mean(0)
		adversarial_attacks.append(adversarial_attack)

	adversarial_attacks = torch.cat(adversarial_attacks)
	return adversarial_attacks

def compute_lrp(x_test, network, rule, device, n_samples, avg_posterior=False):

	x_test = x_test.to(device)

	explanations = []
	for x in tqdm(x_test):

		post_explanations = []

		for idx in range(n_samples):

			# Forward pass
			x_copy = copy.deepcopy(x.detach()).unsqueeze(0)
			x_copy.requires_grad = True

			random.seed(idx)
			y_hat = network.forward(x_copy, explain=True, rule=rule)[0]
			
			# Choose argmax
			y_hat = y_hat[torch.arange(x_copy.shape[0]), y_hat.max(1)[1]]
			y_hat = y_hat.sum()

			# Backward pass (compute explanation)
			y_hat.backward()
			lrp = x_copy.grad.squeeze(1)
			post_explanations.append(lrp)

		post_explanations = torch.stack(post_explanations).mean(0).squeeze()
		explanations.append(post_explanations)

	return torch.stack(explanations) 

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
	"""
	Save the training model
	"""
	torch.save(state, filename)


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, )):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res


if __name__ == '__main__':
	main()

