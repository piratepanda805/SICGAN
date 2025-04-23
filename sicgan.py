import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import List, Tuple, Union

print("CUDA available:", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name(0))

def extract_patches_with_augmentation(
    image: np.ndarray,
    patch_size: Tuple[int, int] = (64, 64),
    num_patches: int = 500,
    min_seed_distance: int = 32
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """
    Extracts a specified number of augmented image patches from a given image.

    Parameters:
    - image (np.ndarray): Input image from which patches are to be extracted.
    - patch_size (Tuple[int, int]): Size (width, height) of each patch.
    - num_patches (int): Total number of patches to return (after augmentation).
    - min_seed_distance (int): Minimum pixel distance between two patch centers to avoid overlap.

    Returns:
    - Tuple containing:
        - List[np.ndarray]: A list of image patches including augmentations.
        - List[Tuple[int, int]]: A list of (x, y) coordinates indicating
          the top-left corner of where each patch (augmented or not) was extracted from.
    """

    # Get image dimensions
    h, w, _ = image.shape

    raw_patches: List[np.ndarray] = []       # To store the original (non-augmented) patches
    seed_points: List[Tuple[int, int]] = []  # To store coordinates of patch origins to enforce distance constraint
    patch_origins: List[Tuple[int, int]] = []  # To track the origin of each raw patch

    # Check if the patch fits within the image boundaries
    def is_within_bounds(x: int, y: int, ph: int, pw: int) -> bool:
        return 0 <= x <= w - pw and 0 <= y <= h - ph

    # Extract a patch at coordinates (x, y)
    def get_patch(x: int, y: int) -> np.ndarray:
        return image[y:y + patch_size[1], x:x + patch_size[0]].copy()

    # Create augmented versions of a patch:
    # - vertical flip, horizontal flip, and 90/180/270-degree rotations
    def augment_patch(patch: np.ndarray) -> List[np.ndarray]:
        aug_patches = [patch]
        aug_patches.append(cv2.flip(patch, 0))  # Vertical flip
        aug_patches.append(cv2.flip(patch, 1))  # Horizontal flip
        for k in [1, 2, 3]:  # Rotate patch 90, 180, and 270 degrees
            aug_patches.append(np.rot90(patch, k))
        return aug_patches

    # Ensure the new patch is not too close to any previously sampled patch
    def is_far_enough(x: int, y: int) -> bool:
        for sx, sy in seed_points:
            if np.hypot(sx - x, sy - y) < min_seed_distance:
                return False
        return True

    # Since each patch will generate 6 augmentations (original + 5),
    # we need fewer raw patches to reach `num_patches`
    max_raw_patches = (num_patches + 5) // 6  # Round up to cover edge cases
    attempts = 0
    max_attempts = max_raw_patches * 20  # Prevent infinite loops in case of poor sampling conditions

    # Keep trying to sample valid patches until we have enough or exceed the attempt limit
    while len(raw_patches) < max_raw_patches and attempts < max_attempts:
        x = random.randint(0, w - patch_size[0])  # Random top-left x
        y = random.randint(0, h - patch_size[1])  # Random top-left y

        if is_within_bounds(x, y, patch_size[0], patch_size[1]) and is_far_enough(x, y):
            patch = get_patch(x, y)
            raw_patches.append(patch)
            seed_points.append((x, y))
            patch_origins.append((x, y))

        attempts += 1

    # Once we have enough raw patches, augment each one
    augmented_patches: List[np.ndarray] = []
    augmented_origins: List[Tuple[int, int]] = []

    for i, patch in enumerate(raw_patches):
        origin = patch_origins[i]
        augmented = augment_patch(patch)  # Get list of 6 versions
        augmented_patches.extend(augmented)  # Add all augmented versions
        augmented_origins.extend([origin] * len(augmented))  # Track their common origin

    # Return only as many patches as requested, in case we went slightly over
    return augmented_patches[:num_patches], augmented_origins[:num_patches]

def prepare_input_output_pairs(
    patches: List[np.ndarray],
    origins: List[Tuple[int, int]],
    patch_size: Tuple[int, int] = (64, 64)
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepares input-output pairs from image patches for supervised learning.

    For each patch, input features include:
    - Grayscale intensity (L channel from LAB)
    - Sobel edge magnitude (based on L)
    - Laplacian of L (edge density)
    - Normalised x and y spatial coordinates within the patch

    The output label for each pixel is:
    - A and B (color opponent channels from LAB)

    Parameters:
    - patches (List[np.ndarray]): List of RGB image patches.
    - origins (List[Tuple[int, int]]): List of (x, y) coordinates for each patch's top-left corner.
    - patch_size (Tuple[int, int]): Width and height of each patch (default: (64, 64)).

    Returns:
    - Tuple[np.ndarray, np.ndarray]:
        - input_features: shape (N, H, W, 5) → 5 features per pixel (L, Sobel, Laplacian, x, y)
        - output_labels: shape (N, H, W, 2) → 2 labels per pixel (A, B)
    """

    input_features: List[np.ndarray] = []
    output_labels: List[np.ndarray] = []

    for patch, (x0, y0) in zip(patches, origins):
        # Convert RGB patch to LAB color space
        lab_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB)

        # Extract individual LAB channels
        L = lab_patch[:, :, 0] # Lightness
        A = lab_patch[:, :, 1]  # Green–Red
        B = lab_patch[:, :, 2]  # Blue–Yellow

        # Compute edge magnitude using Sobel operator on the L channel
        sobel_x = cv2.Sobel(L, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(L, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        # Compute edge density using Laplacian operator on the L channel
        laplacian = cv2.Laplacian(L, cv2.CV_64F)

        # Generate normalised spatial coordinates
        local_x_coords, local_y_coords = np.meshgrid(
            np.arange(patch_size[0]),
            np.arange(patch_size[1])
        )
        local_norm_x = local_x_coords / float(patch_size[0])
        local_norm_y = local_y_coords / float(patch_size[1])

        # Stack features: L, sobel, laplacian, x, y
        input_patch = np.stack([L, sobel_magnitude, laplacian, local_norm_x, local_norm_y], axis=-1)
        input_features.append(input_patch)

        # Labels are the A and B channels
        output_labels.append(np.stack([A, B], axis=-1))

    return np.array(input_features), np.array(output_labels)


class Generator(nn.Module):
    """
    Generator model for predicting A and B channels from 5-channel input:
    V channel, Sobel magnitude, Laplacian, and 2D spatial coordinates.
    """
    def __init__(self) -> None:
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(5, 64, kernel_size=3, stride=1, padding=1),  # Input: 5 channels
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 2, kernel_size=3, stride=1, padding=1),  # Output: 2 channels (A and B)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Discriminator(nn.Module):
    """
    Discriminator model for distinguishing real vs fake (A, B) patches.
    Input: 2-channel image (A, B).
    Output: Scalar score (higher = more real).
    """
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),  # Or (4, 4) or (1, 1)
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class GAN(nn.Module):
    """
    GAN wrapper model combining Generator and Discriminator.
    Runs the generator and evaluates output using the discriminator.
    """
    def __init__(self, generator: Generator, discriminator: Discriminator) -> None:
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        generated_data = self.generator(x)
        return self.discriminator(generated_data)


def add_instance_noise(images: torch.Tensor, std: float = 0.1) -> torch.Tensor:
    """
    Adds instance noise to the input tensor.

    Args:
        images (Tensor): Input image batch.
        std (float): Standard deviation of the Gaussian noise.

    Returns:
        Tensor: Noisy image batch.
    """
    return images + torch.randn_like(images) * std

def train_step(
    generator: nn.Module,
    discriminator: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    generator_optimiser: optim.Optimizer,
    discriminator_optimiser: optim.Optimizer,
    device: torch.device
    ) -> tuple[float, float]:
    """
    Performs one training step of the GAN.

    Args:
        generator (nn.Module): Generator network.
        discriminator (nn.Module): Discriminator network.
        inputs (Tensor): Input features (e.g. V, edge maps, coords).
        targets (Tensor): Ground truth H and S channels.
        generator_optimiser (Optimiser): Optimiser for the generator.
        discriminator_optimiser (Optimiser): Optimiser for the discriminator.
        device (torch.device): Device to run computations on.

    Returns:
        tuple[float, float]: Discriminator loss and generator loss.
    """
    generator.train()
    discriminator.train()

    batch_size = inputs.size(0)

    # Label smoothing and noise
    real_labels = torch.full((batch_size, 1), 0.9, device=device) + 0.05 * torch.rand((batch_size, 1), device=device)
    fake_labels = 0.05 * torch.rand((batch_size, 1), device=device)

    # Generator forward
    fake_HS = generator(inputs)

    # Discriminator predictions with instance noise
    real_preds = discriminator(add_instance_noise(targets))
    fake_preds = discriminator(add_instance_noise(fake_HS.detach()))

    # Losses
    real_loss = criterion(real_preds, real_labels)
    fake_loss = criterion(fake_preds, fake_labels)
    disc_loss = real_loss + fake_loss

    discriminator_optimiser.zero_grad()
    disc_loss.backward()
    discriminator_optimiser.step()

    # Train Generator to fool Discriminator
    gen_labels = torch.full((batch_size, 1), 1.0, device=device) + 0.05 * torch.rand((batch_size, 1), device=device)
    fake_preds = discriminator(fake_HS)
    gen_loss = criterion(fake_preds, gen_labels)

    generator_optimiser.zero_grad()
    gen_loss.backward()
    generator_optimiser.step()

    return disc_loss.item(), gen_loss.item()



def train_gan(
    generator: nn.Module,
    discriminator: nn.Module,
    dataloader: DataLoader,
    epochs: int,
    generator_optimiser: optim.Optimizer,
    discriminator_optimiser: optim.Optimizer,
    device: torch.device
    ) -> None:
    """
    Trains the GAN for a given number of epochs.

    Args:
        generator (nn.Module): Generator model.
        discriminator (nn.Module): Discriminator model.
        dataloader (DataLoader): DataLoader yielding (input, target) pairs.
        epochs (int): Number of training epochs.
        generator_optimiser (Optimiser): Optimiser for the generator.
        discriminator_optimiser (Optimiser): Optimiser for the discriminator.
        device (torch.device): Device for training (CPU or CUDA).
    """
    for epoch in range(epochs):
        print(f"Epoch [{epoch+1}/{epochs}] starting...")
        for i, (input_patches, real_HS) in enumerate(dataloader):
            input_patches = input_patches.to(device)
            real_HS = real_HS.to(device)

            disc_loss, gen_loss = train_step(
                generator,
                discriminator,
                input_patches,
                real_HS,
                generator_optimiser,
                discriminator_optimiser,
                device
            )

            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(dataloader)}], D Loss: {disc_loss:.4f}, G Loss: {gen_loss:.4f}")

        print(f"Epoch [{epoch+1}/{epochs}] completed")


img = cv2.imread("jcsmr.jpg")  # Remember OpenCV loads in BGR
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB if needed

patchesList, patchOrigins = extract_patches_with_augmentation(img, num_patches=5000, patch_size=(128, 128))
input_data, output_data = prepare_input_output_pairs(patchesList, patchOrigins, patch_size=(128, 128))


print("Input shape:", input_data.shape)
print("Output shape:", output_data.shape)

# Optimisers
generator = Generator()
discriminator = Discriminator()

generator_optimiser = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
discriminator_optimiser = torch.optim.Adam(discriminator.parameters(), lr=4e-4, betas=(0.5, 0.999))

criterion = nn.BCEWithLogitsLoss()

input_data_tensor = torch.tensor(input_data, dtype=torch.float32).permute(0, 3, 1, 2)  # Convert to (batch, channels, height, width)
output_data_tensor = torch.tensor(output_data, dtype=torch.float32).permute(0, 3, 1, 2)  # Convert to (batch, channels, height, width)

# Create a dataset and dataloader
dataset = TensorDataset(input_data_tensor, output_data_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Set the device to either GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move models to the correct device
generator = generator.to(device)
discriminator = discriminator.to(device)

# Train the model
epochs = 20
model = train_gan(generator, discriminator, dataloader, epochs, generator_optimiser, discriminator_optimiser, device)


def predict_patch(
    generator: nn.Module,
    patch: np.ndarray,
    patch_origin: Tuple[int, int],
    device: torch.device
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predicts the A and B channels of a LAB patch using the generator model.

    Args:
        generator (nn.Module): Trained generator model.
        patch (np.ndarray): RGB image patch (shape: [H, W, 3]).
        patch_origin (Tuple[int, int]): (x, y) coordinates of the patch origin.
        device (torch.device): Torch device to run the model on.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Predicted A and B channels (float32).
    """
    patch_lab = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB)
    L, A, B = patch_lab[:, :, 0], patch_lab[:, :, 1], patch_lab[:, :, 2]

    # Edge features from L
    sobel_x = cv2.Sobel(L, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(L, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    laplacian = cv2.Laplacian(L, cv2.CV_64F)

    # Global coordinates
    patch_h, patch_w = patch.shape[:2]
    x0, y0 = patch_origin
    local_x_coords, local_y_coords = np.meshgrid(np.arange(patch_w), np.arange(patch_h))
    local_norm_x = local_x_coords / float(patch_w)
    local_norm_y = local_y_coords / float(patch_h)

    # Stack input features
    input_features = np.stack([L, sobel_magnitude, laplacian, local_norm_x, local_norm_y], axis=-1)
    input_tensor = torch.tensor(input_features, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    # Model inference
    with torch.no_grad():
        output = generator(input_tensor).squeeze(0).cpu().numpy()  # shape: (2, H, W)

    predicted_A = output[0]
    predicted_B = output[1]

    return predicted_A, predicted_B

def predict_image(
    generator: nn.Module,
    image: np.ndarray,
    patch_size: Tuple[int, int] = (64, 64),
    device: Union[str, torch.device] = 'cpu'
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predicts the full A and B channels of an RGB image using the generator model patch-by-patch.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Predicted A and B channels (float32).
    """
    h, w, _ = image.shape
    output_A = np.zeros((h, w), dtype=np.float32)
    output_B = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h, patch_size[1]):
        for x in range(0, w, patch_size[0]):
            patch = image[y:y + patch_size[1], x:x + patch_size[0]]

            if patch.shape[0] != patch_size[1] or patch.shape[1] != patch_size[0]:
                continue

            predicted_A, predicted_B = predict_patch(generator, patch, (x, y), device)

            output_A[y:y + patch_size[1], x:x + patch_size[0]] = predicted_A
            output_B[y:y + patch_size[1], x:x + patch_size[0]] = predicted_B

    return output_A, output_B


# Run prediction
pred_A, pred_B = predict_image(generator, img, device=device)

# Use original L channel from LAB
lab_original = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
L_original = lab_original[:, :, 0]

# Merge and convert back to RGB
predicted_lab = cv2.merge([
    L_original.astype(np.uint8), 
    np.clip(pred_A, 0, 255).astype(np.uint8), 
    np.clip(pred_B, 0, 255).astype(np.uint8)
])
predicted_rgb = cv2.cvtColor(predicted_lab, cv2.COLOR_LAB2RGB)

# Show the result
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(img)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Predicted A and B (L original)")
plt.imshow(predicted_rgb)
plt.axis('off')
plt.tight_layout()
plt.savefig("predicted_output_lab.png")
print("Saved result to predicted_output_lab.png")
plt.show()