from dmcnn_vd import DMCNN_VD
from utils import device, best_early_stopping_model_path
from getting_and_init_the_data import get_all_data_loaders
from torch.optim import Adam
from forward_backward_pass import forward_backward_pass_dmcnn_vd
import pickle
from pathlib import Path
import torch
import sys
from pytorch_model_summary import summary


def main():
    # Print out the device
    print(f'Process on {device}', '\n\n')

    # Check the CUDA details
    if torch.cuda.is_available():
        print(f'Device name: {torch.cuda.get_device_name(0)}', '\n\n')

    # Get the job index and set the seed
    job_idx = int(sys.argv[1])
    print('Training DMCNN_VD', '\n\n')

    # Get the model
    dmcnn_vd_network = DMCNN_VD(num_hiddel_blocks=20).to(device)

    # Model summary
    print(summary(dmcnn_vd_network, torch.rand(64,3,33,33).float().to(device)))

    # Optimizer
    optimizer = Adam(dmcnn_vd_network.parameters(), lr=1e-3)

    # Hyperparameters
    batch_size=32
    epochs=500

    # Early stopping criterion
    lowest_val_loss = 1e10
    patience = 30
    best_epoch = 0

    # Data
    train_data_loader, val_data_loader, test_data_loader = get_all_data_loaders(batch_size)

    # Epoch numbers
    epoch_train_loss = []
    epoch_train_mean_psnr = []
    epoch_val_loss = []
    epoch_val_mean_psnr = []

    # Loop over epochs
    for epoch in range(epochs):
        # Train
        dmcnn_vd_network.train()
        dmcnn_vd_network, train_loss, train_mean_psnr, _, _, _, _ = forward_backward_pass_dmcnn_vd(
            dmcnn_vd_network=dmcnn_vd_network,
            data_loader=train_data_loader,
            optimizer=optimizer,
            device=device
        )
        epoch_train_loss.append(train_loss)
        epoch_train_mean_psnr.append(train_mean_psnr)

        # Validation
        with torch.no_grad():
            dmcnn_vd_network.eval()
            _, val_loss, val_mean_psnr, _, _, _, _ = forward_backward_pass_dmcnn_vd(
                dmcnn_vd_network=dmcnn_vd_network,
                data_loader=val_data_loader,
                optimizer=None,
                device=device
            )
        epoch_val_loss.append(val_loss)
        epoch_val_mean_psnr.append(val_mean_psnr)

        print(f"Epoch {epoch} | train loss: {train_loss:.4f} | train mean PSNR: {train_mean_psnr:.4f} | val loss: {val_loss:.4f} | val mean PSNR: {val_mean_psnr:.4f}")

        # Early stopping
        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            best_epoch = epoch
            torch.save(dmcnn_vd_network.state_dict(), Path(best_early_stopping_model_path, f"#{job_idx}_dmcnn_vd_best_model.pt"))
        elif epoch - best_epoch > patience:
            print(f"Early stopping at epoch {best_epoch}, with lowest validation loss of {lowest_val_loss:.4f}")
            break

    # Test
    # Load the best model
    with torch.no_grad():
        dmcnn_vd_network.load_state_dict(torch.load(Path(best_early_stopping_model_path, f"#{job_idx}_dmcnn_vd_best_model.pt")))
        print(f"Loaded the best model from {Path(best_early_stopping_model_path, f'#{job_idx}_dmcnn_vd_best_model.pt')}")
        dmcnn_vd_network.eval()
        _, test_loss, test_mean_psnr, test_mean_channels_psnr, test_img_targets, test_img_demosaiced, test_img_mosaic = forward_backward_pass_dmcnn_vd(
            dmcnn_vd_network=dmcnn_vd_network,
            data_loader=test_data_loader,
            optimizer=None,
            device=device
        )

    print(f"Test loss: {test_loss:.4f} | test mean PSNR: {test_mean_psnr:.4f} | test mean channels PSNR: {test_mean_channels_psnr}")

    # Save the results to pickle file
    result_dict = {
        "epoch_train_loss": epoch_train_loss,
        "epoch_train_mean_psnr": epoch_train_mean_psnr,
        "test_loss": test_loss,
        "test_mean_psnr": test_mean_psnr,
        "test_mean_channels_psnr": test_mean_channels_psnr,
        "test_img_targets": test_img_targets,
        "test_img_demosaiced": test_img_demosaiced,
        "test_img_mosaic": test_img_mosaic
    }

    with open(f"narvi_pickles/#{job_idx}_dmcnn_vd_results.pickle", "wb") as f:
        print('\n\n', f"Saving results to narvi_pickles/#{job_idx}_dmcnn_vd_results.pickle")
        pickle.dump(result_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
    

