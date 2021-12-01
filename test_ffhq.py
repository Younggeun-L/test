import argparse
import numpy as np
import torch
import os
import random
from co_mod_gan import Generator
from PIL import Image
import cv2
import time


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--checkpoint', help='Network checkpoint path', default="checkpoints/co-mod-gan-ffhq-9-025000.pth")
parser.add_argument('-i', '--image', help='Original image path', default = 'input_ffhq')
parser.add_argument('-m', '--mask', help='Mask path', default = 'mask_ffhq')
parser.add_argument('-im', '--maskinput', help='Mask path', default = 'mask_input')
parser.add_argument('-o', '--output', help='Output (inpainted) image path', default = 'ffhq_output')
parser.add_argument('-t', '--truncation', help='Truncation psi for the trade-off between quality and diversity. Defaults to 1.', default=None)
parser.add_argument('--device', help='cpu|cuda', default='cuda')
args = parser.parse_args()

assert args.truncation is None

device = torch.device(args.device)

image_files = os.listdir(args.image)
image_files.sort()
mask_files = os.listdir(args.mask)
mask_files.sort()

#Input 이미지와 마스크를 comodgan을 통해 holepainting 진행

if args.output is not None:
    os.makedirs(args.output, exist_ok=True)

if args.maskinput is not None:
    os.makedirs(args.maskinput, exist_ok=True)

time0 = time.time()
for j in range (0, len(image_files)):

    print("Processing ------------", j+1 , "image")
    
    img = Image.open(args.image + '/' + image_files[j])
    w,h = img.size[:2]
    img = img.resize((512,512))

    #마스크 폴더에서 랜덤으로 마스크 추출
    k = random.randint(0,len(mask_files)-1)
    mask = Image.open(args.mask + '/' + mask_files[k])
    mask = mask.resize((512,512))    

    #mask + input 저장
    image2 = cv2.imread(args.image + '/' + image_files[j])
    image2 = cv2.resize(image2,(512,512))
    mask2 = cv2.imread(args.mask + '/' + mask_files[k])
    mask2 = cv2.resize(mask2,(512,512))
    mask2 = 255-mask2
    mask_image = cv2.add(image2, mask2)
    mask_image = cv2.resize(mask_image,(w,h))
    cv2.imwrite(args.maskinput + '/' + str(image_files[j]), mask_image)

    real = np.asarray(img).transpose([2, 0, 1])/255.0
    masks = np.asarray(mask.convert('1'), dtype=np.float32)

    images = torch.Tensor(real.copy())[None,...]*2-1
    masks = torch.Tensor(masks)[None,None,...].float()
    masks = (masks>0).float()
    latents_in = torch.randn(1, 512)

    net = Generator()
    net.load_state_dict(torch.load(args.checkpoint))
    net.eval()

    net = net.to(device)
    images = images.to(device)
    masks = masks.to(device)
    latents_in = latents_in.to(device)

    result = net(images, masks, [latents_in], truncation=args.truncation)
    result = result.detach().cpu().numpy()
    result = (result+1)/2
    result = (result[0].transpose((1,2,0)))*255
    
    result = Image.fromarray(result.clip(0,255).astype(np.uint8))
    result = result.resize((w,h))
    result.save(args.output + '/' + str(image_files[j]))
    print("Success ---------------", j+1 , "image")

print("time: ", time.time() - time0)