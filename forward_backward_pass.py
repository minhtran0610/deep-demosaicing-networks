import torch
import numpy as np


def psnr(img_target, img_demosaiced):
    mse = torch.mean((img_target - img_demosaiced) ** 2)
    cpsnr = 20 * torch.log10(255.0 / torch.sqrt(mse))
    psnr_channel = []

    for i in range(img_target.shape[0]):
        mse = torch.mean((img_target[i] - img_demosaiced[i]) ** 2)
        psnr_channel.append(20 * torch.log10(255.0 / torch.sqrt(mse)))
    
    return cpsnr, torch.tensor(psnr_channel)


def mean_psnr(img_targets, img_demosaiced):
    cpsnr_values = []
    psnr_channels_values = []

    for i in range(img_targets.shape[0]):
        cpsnr, psnr_channel = psnr(img_targets[i], img_demosaiced[i])
        cpsnr_values.append(cpsnr)
        psnr_channels_values.append(psnr_channel)
    
    return torch.mean(torch.tensor(cpsnr_values)), torch.mean(torch.stack(psnr_channels_values), dim=0)


def forward_backward_pass_dmcnn(dmcnn_network, data_loader, optimizer, device):
    iteration_loss = []

    img_targets = []
    img_mosaic = []
    img_demosaiced = []

    if optimizer is not None:
        dmcnn_network.train()
    else:
        dmcnn_network.eval()

    for batch_idx, (target_img, mosaic_img) in enumerate(data_loader):
        if optimizer is not None:
            optimizer.zero_grad()
        
        target_img = target_img[:, :, 6:27, 6:27].float().to(device)
        mosaic_img = mosaic_img.float().to(device)

        # Demosaic the image
        demosaiced_img = dmcnn_network(mosaic_img)

        # Calculate the loss
        loss = torch.nn.functional.mse_loss(demosaiced_img, target_img)

        if optimizer is not None:
            loss.backward()
            optimizer.step()

        iteration_loss.append(loss.item())
        img_targets.append(target_img)
        img_mosaic.append(mosaic_img)
        img_demosaiced.append(demosaiced_img)

    # Record the predictions
    img_targets = torch.cat(img_targets, dim=0)
    img_mosaic = torch.cat(img_mosaic, dim=0)
    img_demosaiced = torch.cat(img_demosaiced, dim=0)

    # Calculate loss and the mean PSNR
    loss = np.mean(iteration_loss)
    mean_psnr_value, mean_psnr_channels_value = mean_psnr(img_targets, img_demosaiced)

    return dmcnn_network, loss, mean_psnr_value, mean_psnr_channels_value, img_targets.detach().cpu().numpy(), img_demosaiced.detach().cpu().numpy(), img_mosaic.detach().cpu().numpy()
    

def forward_backward_pass_dmcnn_vd(dmcnn_vd_network, data_loader, optimizer, device):
    iteration_loss = []

    img_targets = []
    img_mosaic = []
    img_demosaiced = []

    if optimizer is not None:
        dmcnn_vd_network.train()
    else:
        dmcnn_vd_network.eval()

    for batch_idx, (target_img, mosaic_img) in enumerate(data_loader):
        if optimizer is not None:
            optimizer.zero_grad()
        
        target_img = target_img.float().to(device)
        mosaic_img = mosaic_img.float().to(device)

        # Demosaic the image
        demosaiced_img = dmcnn_vd_network(mosaic_img)

        # Calculate the loss
        loss = torch.nn.functional.mse_loss(demosaiced_img, target_img)

        if optimizer is not None:
            loss.backward()
            optimizer.step()

        iteration_loss.append(loss.item())
        img_targets.append(target_img)
        img_mosaic.append(mosaic_img)
        img_demosaiced.append(demosaiced_img)

    # Record the predictions
    img_targets = torch.cat(img_targets, dim=0)
    img_mosaic = torch.cat(img_mosaic, dim=0)
    img_demosaiced = torch.cat(img_demosaiced, dim=0)

    # Calculate loss and the mean PSNR
    loss = np.mean(iteration_loss)
    mean_psnr_value, mean_psnr_channels_value = mean_psnr(img_targets, img_demosaiced)

    return dmcnn_vd_network, loss, mean_psnr_value, mean_psnr_channels_value, img_targets.detach().cpu().numpy(), img_demosaiced.detach().cpu().numpy(), img_mosaic.detach().cpu().numpy()
