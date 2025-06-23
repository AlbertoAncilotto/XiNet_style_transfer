import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms, datasets

def scale(gram_matrix):
    scaled = gram_matrix + 1.1 - torch.min(gram_matrix)
    log = torch.log(scaled)
    return (log - log.mean()) / log.std()

def gram(tensor):
    B, C, H, W = tensor.shape
    x = tensor.view(B, C, H*W)
    x_t = x.transpose(1, 2)
    return  torch.bmm(x-1.0, x_t-1.0) / (C*H*W) #shifted gram matrix

def load_image(path):
    img = cv2.imread(path)
    return img

def total_variation_loss(img, weight):
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
    tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
    return weight*(tv_h+tv_w)/(bs_img*c_img*h_img*w_img)

def show(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img/255).clip(0,1)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(img)
    plt.show()

def saveimg(img, image_path):
    img = img.clip(0, 255)
    cv2.imwrite(image_path, img)

def itot(img, max_size=None, scale=False):
    if (max_size==None):
        itot_t = transforms.ToTensor()
    else:
        H, W, C = img.shape
        image_size = tuple([int((float(max_size) / max([H,W]))*x) for x in [H, W]])
        itot_t = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

    tensor = itot_t(img)
    if scale:
        tensor = tensor.mul(255)
    tensor = tensor.unsqueeze(dim=0)
    return tensor

def ttoi(tensor):
    tensor = tensor.squeeze()
    img = tensor.cpu().numpy()
    img = img.transpose(1, 2, 0)
    return img

def transfer_color(src, dest, mask=None, alpha=1.0):
    src, dest = src.clip(0,255), dest.clip(0,255)
    H,W,_ = src.shape 
    dest = cv2.resize(dest, dsize=(W, H), interpolation=cv2.INTER_LINEAR)
    if mask is not None:
        mask = cv2.resize(mask, dsize=(W, H), interpolation=cv2.INTER_LINEAR)
    
    dest_gray = cv2.cvtColor(dest, cv2.COLOR_BGR2GRAY)
    src_yiq = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)   
    src_yiq[...,0] = src_yiq[...,0]*(1-alpha)+ alpha*dest_gray   
    
    if mask is not None:
        out = cv2.cvtColor(src_yiq, cv2.COLOR_YCrCb2BGR).clip(0,255)
        out[mask<=0.5,:]=dest[mask<=0.5,:]
    else:
        out = cv2.cvtColor(src_yiq, cv2.COLOR_YCrCb2BGR).clip(0,255)
    return out

def plot_loss_hist(c_loss, s_loss, total_loss, title="Loss History"):
    x = [i for i in range(len(total_loss))]
    plt.figure(figsize=[10, 6])
    plt.plot(x, c_loss, label="Content Loss")
    plt.plot(x, s_loss, label="Style Loss")
    plt.plot(x, total_loss, label="Total Loss")
    
    plt.legend()
    plt.xlabel('Every 500 iterations')
    plt.ylabel('Loss')
    plt.title(title)
    plt.show()

class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (*original_tuple, path)
        return tuple_with_path

def _t(data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.tensor(data, requires_grad=False, dtype=torch.float32, device=device)

def _mul(coeffs, image):
    coeffs = coeffs.to(image.device)
    r0 = image[:, 0:1, :, :].repeat(1, 3, 1, 1) * coeffs[:, 0].view(1, 3, 1, 1)
    r1 = image[:, 1:2, :, :].repeat(1, 3, 1, 1) * coeffs[:, 1].view(1, 3, 1, 1)
    r2 = image[:, 2:3, :, :].repeat(1, 3, 1, 1) * coeffs[:, 2].view(1, 3, 1, 1)
    return r0 + r1 + r2

_RGB_TO_YCBCR = _t([[0.257, 0.504, 0.098], [-0.148, -0.291, 0.439], [0.439 , -0.368, -0.071]])
_YCBCR_OFF = _t([0.063, 0.502, 0.502]).view(1, 3, 1, 1)

def rgb2ycbcr(rgb):
    return _mul(_RGB_TO_YCBCR, rgb) + _YCBCR_OFF.to(rgb.device)

def log_losses(batch_count, losses_sum):
    print("========Iteration {}========".format(batch_count))
    for k, v in losses_sum.items():
        print(f"\t{k.capitalize()} Loss:\t{v / batch_count:.4f}")
    total = sum(losses_sum.values())
    print(f"\tTotal Loss:\t{total / batch_count:.4f}")

def save_outputs(batch_count, content_batch, generated_batch, transformer, save_model_path, save_image_path):
    checkpoint_path = save_model_path + f"checkpoint.pth"
    torch.save(transformer.state_dict(), checkpoint_path)
    print(f"Saved TransformerNetwork checkpoint file at {checkpoint_path}")

    output_batch = torch.cat((content_batch, generated_batch), dim=2)
    sample_tensor = output_batch.clone().detach()
    sample_tensor = sample_tensor.permute(1, 2, 0, 3).reshape(3, sample_tensor.size(2), -1)
    sample_image = ttoi(sample_tensor.clone().detach())
    sample_image_path = save_image_path + f"sample_{batch_count}.png"
    saveimg(sample_image, sample_image_path)
    print(f"Saved sample transformed image at {sample_image_path}")
    plt.figure(figsize=(10, 5))
    plt.imshow(sample_image[:,:,::-1]/255.0)
    plt.axis("off")
    plt.show()

def model_info(model, verbose=True, img_size=224, get_flops=False):
    n_p = sum(x.numel() for x in model.parameters())  
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad) 
    if verbose:
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    if get_flops:
        from thop import profile
        from copy import deepcopy
        stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32
        img = torch.zeros((1, 3, stride, stride), device=next(model.parameters()).device) 
        flops = profile(deepcopy(model), inputs=(img.unsqueeze(0)), verbose=True)[0] / 1E6  
        img_size = img_size if isinstance(img_size, list) else [img_size, img_size]  
        mmacc = flops * img_size[0] / stride * img_size[1] / stride / 2
        fs = ', %.1f M MACC' % (mmacc) 


    print(f"Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}, @{img_size}")
    return n_p, mmacc