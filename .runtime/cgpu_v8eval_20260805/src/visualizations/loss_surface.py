import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def visualize_loss_surface(model, data_loader, criterion=None, resolution=15, x_range=(-1, 1), y_range=(-1, 1),
                           title="ResNet50 Loss Surface"):
    """
    Visualize the loss surface of a model using an existing train_loader.

    Args:
        model: target model
        data_loader: existing DataLoader
        criterion: Loss function (defaults to CrossEntropyLoss if None)
        resolution: Number of points to sample along each axis
        x_range: Range for the x direction
        y_range: Range for the y direction
        title: Title for the plot
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # Set to evaluation mode

    # Use provided criterion or default to CrossEntropyLoss
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    # Get a batch of data for evaluation
    inputs, targets = next(iter(data_loader))
    inputs = inputs.to(device)
    targets = targets.to(device)

    # Store original weights
    original_weights = {}
    for name, param in model.named_parameters():
        original_weights[name] = param.data.clone()

    # Generate two random directions for perturbation
    direction1 = {}
    direction2 = {}
    for name, param in model.named_parameters():
        direction1[name] = torch.randn_like(param.data)
        direction2[name] = torch.randn_like(param.data)

        # Normalize the directions
        direction1[name] = direction1[name] / direction1[name].norm()
        direction2[name] = direction2[name] / direction2[name].norm()

    # Create grid for evaluation
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    # Evaluate loss at each grid point
    print("Computing loss landscape...")
    total_points = resolution * resolution
    for i in range(resolution):
        for j in range(resolution):
            # Perturb the weights
            alpha = x[j]
            beta = y[i]

            for name, param in model.named_parameters():
                param.data = original_weights[name] + alpha * direction1[name] + beta * direction2[name]

            # Compute loss
            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                Z[i, j] = loss.item()

            # Print progress
            point_number = i * resolution + j + 1
            if point_number % 50 == 0 or point_number == total_points:
                print(
                    f"Progress: {point_number}/{total_points} points calculated ({(point_number / total_points) * 100:.1f}%)")

    # Restore original weights
    for name, param in model.named_parameters():
        param.data = original_weights[name]

    # Plot the surface
    print("Creating 3D visualization...")
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create surface plot
    surf = ax.plot_surface(X, Y, Z, cmap="coolwarm", linewidth=0, antialiased=True, alpha=0.8)

    # Add a contour plot at the bottom
    # ax.contour(X, Y, Z, zdir='z', offset=np.min(Z), cmap="coolwarm")

    # Add color bar
    fig.colorbar(surf, shrink=0.3, aspect=7)

    # Set labels and title
    ax.set_xlabel('Direction 1')
    ax.set_ylabel('Direction 2')
    ax.set_zlabel('Loss')
    ax.set_title(title)

    # Adjust view angle
    ax.view_init(elev=30, azim=45)

    plt.tight_layout()

    # Save the plot
    filename = 'resnet50_loss_surface.png'
    plt.savefig(filename, dpi=300)
    print(f"Plot saved as {filename}")

    # Show the plot
    plt.show()
