import os
import nni
import copy
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils
import utils
from model import NetworkCIFAR as Network
import genotypes
from data_loader import get_cifar
import pandas as pd
from model_factory import create_cnn_model, is_resnet


def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	else:
		return False
	
	
def parse_arguments():
	parser = argparse.ArgumentParser(description='TA Knowledge Distillation Code')
	parser.add_argument('--epochs', default=160, type=int,  help='number of total epochs to run')
	parser.add_argument('--dataset', default='cifar10', type=str, help='dataset. can be either cifar10 or cifar100')
	parser.add_argument('--batch_size', default=80, type=int, help='batch_size')
	parser.add_argument('--learning-rate', default=0.1, type=float, help='initial learning rate')
	parser.add_argument('--momentum', default=0.9, type=float,  help='SGD momentum')
	parser.add_argument('--weight-decay', default=1e-4, type=float, help='SGD weight decay (default: 1e-4)')
	parser.add_argument('--teacher', default='resnet8', type=str, help='teacher student name')
	parser.add_argument('--student', '--model', default='resnet8', type=str, help='teacher student name')
	parser.add_argument('--teacher-checkpoint', default='resnet8_02_best.pth.tar', type=str, help='optinal pretrained checkpoint for teacher')
	parser.add_argument('--cuda', default=1, type=str2bool, help='whether or not use cuda(train on GPU)')
	parser.add_argument('--dataset-dir', default='./data', type=str,  help='dataset directory')
	parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
	args = parser.parse_args()
	return args


def load_checkpoint(model, checkpoint_path):
	"""
	Loads weights from checkpoint
	:param model: a pytorch nn student
	:param str checkpoint_path: address/path of a file
	:return: pytorch nn student with weights loaded from checkpoint
	"""
	model_ckp = torch.load(checkpoint_path)
	model.load_state_dict(model_ckp['model_state_dict'])
	return model


class TrainManager(object):
	def __init__(self, student, teacher=None, train_loader=None, test_loader=None, train_config={}):
		self.student = student
		self.teacher = teacher
		self.have_teacher = bool(self.teacher)
		self.device = train_config['device']
		self.name = train_config['name']
		self.optimizer = optim.SGD(self.student.parameters(),
								   lr=train_config['learning_rate'],
								   momentum=train_config['momentum'],
								   weight_decay=train_config['weight_decay'])
		if self.have_teacher:
			self.teacher.eval()
			self.teacher.train(mode=False)
			
		self.train_loader = train_loader
		self.test_loader = test_loader
		self.config = train_config
	
	def train(self):
		lambda_ = self.config['lambda_student']
		T = self.config['T_student']
		epochs = self.config['epochs']
		trial_id = self.config['trial_id']
		
		max_val_acc = 0
		iteration = 0
		best_acc = 0
		criterion = nn.CrossEntropyLoss()
		training_process = pd.DataFrame(columns=['epochs', 'acc'])

		for epoch in range(epochs):
			self.student.train()
			self.adjust_learning_rate(self.optimizer, epoch)
			loss = 0
			print('-------------')
			max_lenth = len(self.train_loader)
			for batch_idx, (data, target) in enumerate(self.train_loader):
				iteration += 1
				data = data.to(self.device)
				target = target.to(self.device)
				self.optimizer.zero_grad()
				if args.student == 'DARTS':
					output, _ = self.student(data)
				else:
					output = self.student(data)
				print('current epoch: ', str(epoch), 'current process:', str(batch_idx/max_lenth))
				# Standard Learning Loss ( Classification Loss)
				loss_SL = criterion(output, target)
				loss = loss_SL

				if self.have_teacher:
					if args.teacher == 'DARTS':
						teacher_outputs, _ = self.teacher(data)
					else:
						teacher_outputs = self.teacher(data)
					# Knowledge Distillation Loss
					loss_KD = nn.KLDivLoss()(F.log_softmax(output / T, dim=1), F.softmax(teacher_outputs / T, dim=1))
					loss = (1 - lambda_) * loss_SL + lambda_ * T * T * loss_KD

				loss.backward()
				self.optimizer.step()
			
			print("epoch {}/{}".format(epoch, epochs))
			val_acc = self.validate(step=epoch)
			training_process.append([{'epochs': epoch, 'acc': val_acc}])

			if val_acc > best_acc:
				best_acc = val_acc
				self.save(epoch, name='{}_{}_best.pth.tar'.format(self.name, trial_id))
		
		return best_acc, training_process
	
	def validate(self, step=0):
		self.student.eval()
		with torch.no_grad():
			correct = 0
			total = 0
			acc = 0
			for images, labels in self.test_loader:
				images = images.to(self.device)
				labels = labels.to(self.device)
				if args.student == 'DARTS':
					outputs, _ = self.student(images)
				else:
					outputs = self.student(images)
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()
			# self.accuracy_history.append(acc)
			acc = 100 * correct / total
			
			print('{{"metric": "{}_val_accuracy", "value": {}}}'.format(self.name, acc))
			return acc
	
	def save(self, epoch, name=None):
		trial_id = self.config['trial_id']
		if name is None:
			torch.save({
				'epoch': epoch,
				'model_state_dict': self.student.state_dict(),
				'optimizer_state_dict': self.optimizer.state_dict(),
			}, '{}_{}_epoch{}.pth.tar'.format(self.name, trial_id, epoch))
		else:
			torch.save({
				'model_state_dict': self.student.state_dict(),
				'optimizer_state_dict': self.optimizer.state_dict(),
				'epoch': epoch,
			}, name)
	
	def adjust_learning_rate(self, optimizer, epoch):
		epochs = self.config['epochs']
		models_are_plane = self.config['is_plane']
		
		# depending on dataset
		if models_are_plane:
			lr = 0.01
		else:
			if epoch < int(epoch/2.0):
				lr = 0.1
			elif epoch < int(epochs*3/4.0):
				lr = 0.1 * 0.1
			else:
				lr = 0.1 * 0.01
		
		# update optimizer's learning rate
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr


if __name__ == "__main__":

	layer = 9
	# Parsing arguments and prepare settings for training
	args = parse_arguments()
	print(args)
	config = nni.get_next_parameter()
	config = {"lambda_student": 0.5, "T_student": 5, "seed": 20}

	torch.manual_seed(config['seed'])
	torch.cuda.manual_seed(config['seed'])
	# trial_id = os.environ.get('NNI_TRIAL_JOB_ID')
	trial_id = '02'
	dataset = args.dataset
	num_classes = 100 if dataset == 'cifar100' else 'cifar10'
	teacher_model = None
	if args.student == 'DARTS':
		genotype = eval("genotypes.%s" % args.arch)
		student_model = Network(36, 10, layer, True, genotype)
		student_model.cuda()
		# utils.load(student_model, 'cifar10_model.pt')
		student_model.drop_path_prob = 0.2
	else:
		student_model = create_cnn_model(args.student, dataset, use_cuda=args.cuda)


	train_config = {
		'epochs': args.epochs,
		'learning_rate': args.learning_rate,
		'momentum': args.momentum,
		'weight_decay': args.weight_decay,
		'device': 'cuda' if args.cuda else 'cpu',
		'is_plane': not is_resnet(args.student),
		'trial_id': trial_id,
		'T_student': config.get('T_student'),
		'lambda_student': config.get('lambda_student'),
	}
	
	# Train Teacher if provided a teacher, otherwise it's a normal training using only cross entropy loss
	# This is for training single models(NOKD in paper) for baselines models (or training the first teacher)
	if args.teacher:
		if args.teacher == 'DARTS':
			genotype = eval("genotypes.%s" % args.arch)
			teacher_model = Network(36, 10, layer, True, genotype)
			teacher_model.cuda()
			teacher_model.drop_path_prob = 0.2
		else:
			teacher_model = create_cnn_model(args.teacher, dataset, use_cuda=args.cuda)
		if args.teacher_checkpoint:
			print("---------- Loading Teacher -------")
			teacher_model = load_checkpoint(teacher_model, args.teacher_checkpoint)
			train_loader, test_loader = get_cifar(num_classes)
			teacher_train_config = copy.deepcopy(train_config)
			teacher_train_config['name'] = args.teacher
			teacher_trainer = TrainManager(teacher_model, teacher=None, train_loader=train_loader,test_loader=test_loader, train_config=teacher_train_config)
			acc = teacher_trainer.validate()
			print('Teacher ACC:', acc)

		else:
			print("---------- Training Teacher -------")
			train_loader, test_loader = get_cifar(num_classes)
			teacher_train_config = copy.deepcopy(train_config)
			teacher_name = '{}_{}_best.pth.tar'.format(args.teacher, trial_id)
			teacher_train_config['name'] = args.teacher
			teacher_trainer = TrainManager(teacher_model, teacher=None, train_loader=train_loader, test_loader=test_loader, train_config=teacher_train_config)
			best_teacher_acc, process_form = teacher_trainer.train()
			teacher_model = load_checkpoint(teacher_model, os.path.join('./', teacher_name))
			
	# Student training

	print("Teacher param size = %fMB", utils.count_parameters_in_MB(teacher_model))
	print("Student param size = %fMB", utils.count_parameters_in_MB(student_model))

	print("---------- Training Student -------")
	student_train_config = copy.deepcopy(train_config)
	train_loader, test_loader = get_cifar(num_classes, batch_size=args.batch_size)
	student_train_config['name'] = args.student
	student_trainer = TrainManager(student_model, teacher=teacher_model, train_loader=train_loader, test_loader=test_loader, train_config=student_train_config)
	best_student_acc, process_form = student_trainer.train()
	process_form.to_csv(args.student+'_'+train_config['trial_id']+'.csv')
	nni.report_final_result(best_student_acc)
