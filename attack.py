import os
import sys
import numpy as np

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

from models import *
from utils import progress_bar
from utils import plot_image
from torch.autograd import Variable

from differential_evolution import differential_evolution

parser = argparse.ArgumentParser(description='One pixel attack with PyTorch')
parser.add_argument('--model', default='vgg16', help='The target model')
parser.add_argument('--pixels', default=1, type=int, help='The number of pixels that can be perturbed.')
parser.add_argument('--maxiter', default=100, type=int, help='The maximum number of iteration in the DE algorithm.')
parser.add_argument('--popsize', default=400, type=int, help='The number of adverisal examples in each iteration.')
parser.add_argument('--samples', default=100, type=int, help='The number of image samples to attack.')
parser.add_argument('--targeted', action='store_true', help='Set this switch to test for targeted attacks.')
parser.add_argument('--save', default='./results/results.pkl', help='Save location for the results with pickle.')
parser.add_argument('--verbose', action='store_true', help='Print out additional information every iteration.')

args = parser.parse_args()

def imshow(img):    # unnormalize
	npimg = img.numpy()
	npimg = (npimg - np.min(npimg))/ (np.max(npimg) - np.min(npimg))
	#print(npimg)
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()
	# plt.imsave("filename", np.transpose(npimg, (1, 2, 0)))

def imsave(filename, img):    # unnormalize
	npimg = img.numpy()
	npimg = (npimg - np.min(npimg))/ (np.max(npimg) - np.min(npimg))
	#print(npimg)
	plt.imsave(filename, np.transpose(npimg, (1, 2, 0)))

# xs: attack results, searched perturbations.
# image: image
def perturb_image(xs, img):
	img_origin = img.clone()[0]
	# xs shape: (400, 5)
	if xs.ndim < 2:
		xs = np.array([xs])
	batch = len(xs)			# Size of Population: 400
	imgs = img.repeat(batch, 1, 1, 1)  # imgs: torch.Size([400, 3, 32, 32])
	xs = xs.astype(int)

	count = 0
	for x in xs:							# Iteration of population size
		pixels = np.split(x, len(x)/5)
		
		#print("Before Perturb")
		#imshow(imgs[count])
		#print(torch.max(imgs[count]))
		#print(torch.min(imgs[count]))

		for pixel in pixels:

			x_pos, y_pos, r, g, b = pixel
			# count:batch_size(pop size), channels, x_pos, y_pos
			# RGB sensitivity check ???
			# When we read image value, it needs to keep an original image for reading specific position value
			# To flip bit, New RGB = (RGB value >> specific_bit_position) & 1
			#imgs[count, 0, x_pos, y_pos] = (r/255.0-0.4914)/0.2023
			#imgs[count, 1, x_pos, y_pos] = (g/255.0-0.4822)/0.1994
			#imgs[count, 2, x_pos, y_pos] = (b/255.0-0.4465)/0.2010
			rr = img_origin[0, x_pos, y_pos]
			gg = img_origin[1, x_pos, y_pos]
			bb = img_origin[2, x_pos, y_pos]
			#print("Before:: rr/gg/bb : ", rr, gg, bb)
			#print("r/g/b : ", r, g, b)
			frr = (int((rr*0.2023 + 0.4914)*255) ^ (1 << r))
			fgg = (int((gg*0.1994 + 0.4822)*255) ^ (1 << g))
			fbb = (int((bb*0.2010 + 0.4465)*255) ^ (1 << b))
			#print("frr/fgg/fbb : ", frr, fgg, fbb)
			imgs[count, 0, x_pos, y_pos] = (frr/255.0-0.4914)/0.2023
			imgs[count, 1, x_pos, y_pos] = (fgg/255.0-0.4822)/0.1994
			imgs[count, 2, x_pos, y_pos] = (fbb/255.0-0.4465)/0.2010
			#print("After:: rr/gg/bb : ", imgs[count, 0, x_pos, y_pos], imgs[count, 1, x_pos, y_pos], imgs[count, 2, x_pos, y_pos])

		#print("After Perturb")
		#imshow(imgs[count])
		#print(torch.max(imgs[count]))
		#print(torch.min(imgs[count]))
		count += 1

	return imgs

def predict_classes(xs, img, target_calss, net, minimize=True):
	# perturb_image makes population size of perturbated images.
	imgs_perturbed = perturb_image(xs,			# perturbation
	                               img.clone()	# clone of img to keep original img.
								   )

	#print("predict_classes")
	#imshow(imgs_perturbed[0])
	#input = Variable(imgs_perturbed, volatile=True).cuda()
	input = Variable(imgs_perturbed).cuda()

	# net(input) shape : [400, 10], where 400 is pop size.
	# predictions shape: (400,)
	predictions = F.softmax(net(input), dim=1).data.cpu().numpy()[:, target_calss]

	return predictions if minimize else 1 - predictions


def attack_success(x, img, target_calss, net, targeted_attack=False, verbose=False):

	attack_image = perturb_image(x, img.clone())
	#input = Variable(attack_image, volatile=True).cuda()
	input = Variable(attack_image).cuda()
	#confidence = F.softmax(net(input)).data.cpu().numpy()[0]
	confidence = F.softmax(net(input), dim=1).data.cpu().numpy()  ## cpu().numpy() == .item() ??

	predicted_class = np.argmax(confidence)

	if (verbose):
		print("Confidence: %.4f"%confidence[target_calss])
	if (targeted_attack and predicted_class == target_calss) or (not targeted_attack and predicted_class != target_calss):
		return True


def attack(img, label, net, target=None, pixels=1, maxiter=75, popsize=400, verbose=False):
	# img: 1*3*W*H tensor
	# label: a number

	targeted_attack = target is not None
	target_calss = target if targeted_attack else label

	#bounds = [(0,32), (0,32), (0,255), (0,255), (0,255)] * pixels
	bounds = [(0,32), (0,32), (0,7), (0,7), (0,7)] * pixels


	popmul = int(max(1, popsize/len(bounds)))

	predict_fn = lambda xs: predict_classes(				# Probability of Target Class
		xs, img, target_calss, net, target is None)
	callback_fn = lambda x, convergence: attack_success(	# TRUE or FALSE
		x, img, target_calss, net, targeted_attack, verbose)

	# inits shape : [400, 5*pixels]
	inits = np.zeros([popmul*len(bounds), len(bounds)])

	# Make an initial perturbation list
	for init in inits:
		for i in range(pixels):
			init[i*5+0] = np.random.random()*32			# x
			init[i*5+1] = np.random.random()*32			# y
			#init[i*5+2] = np.random.normal(128,127)	# Red
			#init[i*5+3] = np.random.normal(128,127)	# Green
			#init[i*5+4] = np.random.normal(128,127)	# Blue
			init[i*5+2] = np.random.randint(0, 8)		# Red: 0 - 7
			init[i*5+3] = np.random.randint(0, 8)		# Green: 0 - 7
			init[i*5+4] = np.random.randint(0, 8)		# Blue: 0 - 7

	attack_result = differential_evolution(predict_fn, bounds, maxiter=maxiter, popsize=popmul,
		recombination=1, atol=-1, callback=callback_fn, polish=False, init=inits)

	attack_image = perturb_image(attack_result.x, img)

	#attack_var = Variable(attack_image, volatile=True).cuda()
	attack_var = Variable(attack_image).cuda()
	#predicted_probs = F.softmax(net(attack_var)).data.cpu().numpy()[0]
	predicted_probs = F.softmax(net(attack_var), dim=1).data.cpu().numpy()

	predicted_class = np.argmax(predicted_probs)
	print("\nOriginal Class: ", class_names[label], " ==> Predicted Class after Attack: ", class_names[predicted_class], "\n")
	#print("Pertubated Probability: ", predicted_probs)
	if (not targeted_attack and predicted_class != label) or (targeted_attack and predicted_class == target_calss):
		filename = class_names[label] + "_" + class_names[predicted_class] + str(success) + ".png"
		imsave(filename, attack_image[0])
		return 1, attack_result.x.astype(int)
	return 0, [None]



def attack_all(net, loader, pixels=1, targeted=False, maxiter=75, popsize=400, verbose=False):

	global success
	correct = 0
	success = 0

	for batch_idx, (input, target) in enumerate(loader):  # loader : test_loader. Batch=1
		myimg = input[0]  # Batch size = 1. Dim = [1, 3, 32, 32]. Eliminate BATCH dim.
		#imshow(myimg)
		#img_var = Variable(input, volatile=True).cuda()
		img_var = Variable(input).cuda()
		# net(img_var).shape: [1, 10]
		prior_probs = F.softmax(net(img_var), dim=1)
		#print("Original Prob: ", prior_probs)
		_, indices = torch.max(prior_probs, 1)
		
		#print("Original Class: ", indices.data.cpu()[0])
		if target[0] != indices.data.cpu()[0]:
			continue

		correct += 1
		target = target.numpy()

		targets = [None] if not targeted else range(10)

		for target_calss in targets:
			if (targeted):
				if (target_calss == target[0]):
					continue
			# flag: Success 1, Fail 0
			# x: Pertubation when Success
			flag, x = attack(input,				# input image
			                 target[0], 		# label
							 net, 				# network
							 target_calss, 		# target class for attack
							 pixels=pixels, 	# # of pixels
							 maxiter=maxiter, 
							 popsize=popsize, 
							 verbose=verbose)

			success += flag
			if (targeted):
				success_rate = float(success)/(9*correct)
			else:
				success_rate = float(success)/correct

			if flag == 1:
				print("success rate: %.4f (%d/%d) [(x,y) = (%d,%d) and (R,G,B)=(%d,%d,%d)]"%(
					success_rate, success, correct, x[0],x[1],x[2],x[3],x[4]))
		
		if correct == args.samples:
			break

	return success_rate

def main():

	print("==> Loading data and model...")
	tranfrom_test = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])
	test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=tranfrom_test)
	testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=2)

	global class_names
	class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
	assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
	checkpoint = torch.load('./checkpoint/%s.t7'%args.model)
	net = checkpoint['net']
	net.cuda()
	cudnn.benchmark = True

	print("==> Starting attck...")
	print("# of Pixel:", args.pixels, " target (1) / non-target (0):", args.targeted)

	#parser.add_argument('--pixels', default=1, type=int, help='The number of pixels that can be perturbed.')
	#parser.add_argument('--maxiter', default=100, type=int, help='The maximum number of iteration in the DE algorithm.')
	#parser.add_argument('--popsize', default=400, type=int, help='The number of adverisal examples in each iteration.')
	#parser.add_argument('--samples', default=100, type=int, help='The number of image samples to attack.')

	results = attack_all(net, 						# Deep Network
	                     testloader, 				# Test Data Set
						 pixels=args.pixels,		# of Pixels
						 targeted=args.targeted, 
						 maxiter=200, #args.maxiter, 
						 popsize=500, #args.popsize, 
						 verbose=args.verbose)

	print("Final success rate: %.4f"%results)


if __name__ == '__main__':
	main()