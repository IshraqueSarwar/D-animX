import matplotlib.pyplot as plt
import torchvision
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
import numpy as np
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
	# creates a tensor of len-timesteps with values that are evenly spaced from start to end.
	return torch.linspace(start, end, timesteps)


def extract(a,t,x_shape):
	batch_size = t.shape[0]
	out = a.gather(-1, t.cpu())
	return out.reshape(batch_size, *( (1, )*(len(x_shape)-1) )).to(t.device)

# forward diffusion process p(x_{t-1}|x_t)
def q_sample(x_start, t, noise = None):
	if noise is None:
		noise = torch.randn_like(x_start)
	sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
	sqrt_one_minus_alphas_cumprod_t = extract(
		sqrt_one_minus_alpha_comprod, t, x_start.shape
		)
	return sqrt_alphas_cumprod_t*x_start + sqrt_one_minus_alphas_cumprod_t*noise

def get_noisy_image(x_start, t):
	x_noisy = q_sample(x_start, t=t)
	noisy_img = reverse_transform(x_noisy.squeeze())
	return noisy_img


T = 200
betas = linear_beta_schedule(timesteps=T)
alphas = 1.0-betas
alphas_cumprob = torch.cumprod(alphas,axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprob[:-1], (1, 0), value = 1.0)
sqrt_recip_alphas = torch.sqrt(1.0/alphas)

sqrt_alphas_cumprod = torch.sqrt(alphas_cumprob)
sqrt_one_minus_alpha_comprod = torch.sqrt(1.0- alphas_cumprob)

posterior_variance = betas * (1.- alphas_cumprod_prev)/(1. - alphas_cumprob)



# Noise is added to PyTorch tensors, rather than Pillow Images. 
# We'll first define image transformations that allow us to go 
# from a PIL image to a PyTorch tensor (on which we can add the noise), 
# and vice versa.
#converting to torch tensor from PIL image.
image = Image.open('testPic/test.jpg')
image_size = 128
transform = Compose([
	Resize(image_size),
	CenterCrop(image_size),
	ToTensor(), # turn into Numpy array of shape HWC, divide by 255
	# Lambda(lambda t: (t*2)-1)
	])

# converting torch Tensors back to PIL images that can be viewed.
reverse_transform = Compose([
	# Lambda(lambda t: (t+1)/2),
	# Lambda(lambda t: t.permute(1,2,0)),
	# Lambda(lambda t: t*255.),
	# Lambda(lambda t: t.numpy().astype(np.int8)),
	ToPILImage(),
	])

x_start = transform(image).unsqueeze(0)
before = reverse_transform(x_start.squeeze())
#take timesteps 40
t = torch.tensor([10])
after = get_noisy_image(x_start, t)

