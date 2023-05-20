import torch 
import cv2 
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights
from pgd import pgd_attack 
from torch.autograd import Variable

from torchvision import transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Initialize model with the best available weights
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.to(device)
model.eval()


# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Note that the image need to preprocess before feed into the model and reverse to the original range to visualize 
mean = torch.tensor([0.4915, 0.4823, 0.4468])
std = torch.tensor([0.2470, 0.2435, 0.2616])

normalize = transforms.Normalize(mean.tolist(), std.tolist()) 
unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())

# List all available images in the folders 
import glob 

all_images = glob.glob('images/*.jpeg') + glob.glob('images/*.png')

for idx, path in enumerate(all_images): 
    print("Image: {}".format(path))
    img = read_image(path)
    img = img[:3, :, :] # remove alpha channel

    # Step 3: Apply inference preprocessing transforms
    batch = preprocess(img).unsqueeze(0).to(device)

    # Step 4: Use the model and print the predicted category
    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    print(f"{category_name}: {100 * score:.1f}%")

    # Define attack params 
    attack_params = {
        'projecting': True, 
        'random_init': True, 
        'epsilon': 0.031, # 8/255, perturbation size 
        'num_steps': 100, 
        'step_size': 0.007, # 2/255, step size
        'loss_type': 'ce',
        'x_min': batch.min().item(), 
        'x_max': batch.max().item(),
        'y_target': torch.tensor([417]).to(device), # targeted class: ballon, 
        'targeted': True,
    }

    batch = batch.to(device)
    target = torch.tensor([class_id]).to(device)
    X_adv, _ = pgd_attack(model, batch, target, device, attack_params)
    X_adv = Variable(X_adv.data, requires_grad=False)

    perturb = X_adv - batch

    print("Input range")
    print(f"Min: {batch.min().item()}")
    print(f"Max: {batch.max().item()}")
    print("")

    print("Output range")
    print(f"Min: {X_adv.min().item()}")
    print(f"Max: {X_adv.max().item()}")
    print("")

    print("Perturb range")
    print(f"Min: {perturb.min().item()}")
    print(f"Max: {perturb.max().item()}")
    print("")

    prediction = model(X_adv).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    adv_score = prediction[class_id].item()
    adv_category_name = weights.meta["categories"][class_id]
    print(f"{adv_category_name}: {100 * adv_score:.1f}%")

    # Plot the original image and the adversarial image with the perturbation and prediction 
    import matplotlib.pyplot as plt
    import numpy as np

    vmin = 0
    vmax = 255

    # convert to original image range 
    batch = unnormalize(batch)
    # perturb = unnormalize(perturb)
    X_adv = unnormalize(X_adv)

    print("Input range - After unnormalize")
    print(f"Min: {batch.min().item()}")
    print(f"Max: {batch.max().item()}")
    print("")

    print("Output range - After unnormalize")
    print(f"Min: {X_adv.min().item()}")
    print(f"Max: {X_adv.max().item()}")
    print("")

    print("Perturb range - After unnormalize")
    print(f"Min: {perturb.min().item()}")
    print(f"Max: {perturb.max().item()}")
    print("")

    # upscale the perturbation
    k = 10 
    perturb = perturb * k

    # clip to original image range 
    batch = torch.clamp(batch, 0, 1)
    perturb = torch.clamp(perturb, 0, 1)
    X_adv = torch.clamp(X_adv, 0, 1)

    batch = batch[0].permute(1, 2, 0).detach().cpu().numpy() * 255 
    perturb = perturb[0].permute(1, 2, 0).detach().cpu().numpy() * 255
    X_adv = X_adv[0].permute(1, 2, 0).detach().cpu().numpy() * 255



    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(batch.astype(np.uint8), vmin=vmin, vmax=vmax)
    ax[0].set_title('Original Image' + '\n' + '"' + category_name + '"' + 'with {:.1f}% confidence'.format(100 * score))
    ax[0].axis('off')


    ax[1].imshow(perturb.astype(np.uint8), vmin=vmin, vmax=vmax)
    ax[1].set_title('Perturbation' + '\n' + 'Upscale by {}x'.format(k))
    ax[1].axis('off')

    ax[2].imshow(X_adv.astype(np.uint8), vmin=vmin, vmax=vmax)
    ax[2].set_title('Adversarial Image' + '\n' + '"' + adv_category_name + '"' + 'with {:.1f}% confidence'.format(100 * adv_score) )
    ax[2].axis('off')

    plt.show()
    plt.savefig('results/adversarial_{}_{}.png'.format(idx, k), dpi=300)
