# ================================================================
# SINGLE IMAGE + SINGLE SHAPEFILE
# PYTORCH U-NET SEMANTIC SEGMENTATION
# ================================================================
#
# Purpose:
#
#   One satellite image
#        +
#   One polygon shapefile
#        |
#        v
#   Raster segmentation mask
#        |
#        v
#   U-Net training
#        |
#        v
#   Prediction
#
#
# Classes:
#
#   0 = background
#   1 = building
#   2 = road
#   3 = trees
#   4 = land
#
# ================================================================

import os
import warnings

import cv2
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

from rasterio import features

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader


warnings.filterwarnings("ignore")


# ================================================================
# CONFIGURATION
# ================================================================

IMAGE_PATH = (
    "datasets/mandya/images/image_001.tif"
)

SHAPEFILE_PATH = (
    "datasets/mandya/shapefiles/image_001.shp"
)

MODEL_DIR = "models"

OUTPUT_DIR = "outputs"

MODEL_PATH = (
    "models/mandya_unet.pth"
)

# ------------------------------------------------
# Image size
# ------------------------------------------------

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512

IMAGE_SIZE = (
    IMAGE_WIDTH,
    IMAGE_HEIGHT
)

# ------------------------------------------------
# Classes
# ------------------------------------------------

NUM_CLASSES = 5

CLASS_NAMES = {

    0: "background",

    1: "building",

    2: "road",

    3: "trees",

    4: "land"
}

# ------------------------------------------------
# Visualization colors
#
# RGB
# ------------------------------------------------

CLASS_COLORS = {

    0: (0, 0, 0),

    1: (255, 0, 0),

    2: (0, 255, 0),

    3: (0, 128, 0),

    4: (255, 255, 0)
}

# ------------------------------------------------
# Training configuration
# ------------------------------------------------

BATCH_SIZE = 1

EPOCHS = 100

LEARNING_RATE = 1e-4


# ================================================================
# DIRECTORIES
# ================================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ================================================================
# DEVICE
# ================================================================

DEVICE = torch.device(

    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print()
print("================================================")
print("DEVICE")
print("================================================")

print(
    "PyTorch:",
    torch.__version__
)

print(
    "Device:",
    DEVICE
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ================================================================
# STEP 1
# LOAD IMAGE
# ================================================================

def load_satellite_image(
    image_path
):

    print()
    print("================================================")
    print("LOADING SATELLITE IMAGE")
    print("================================================")

    print(
        "Image:",
        image_path
    )

    if not os.path.exists(
        image_path
    ):

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # ------------------------------------------------------------
    # Open GeoTIFF
    # ------------------------------------------------------------

    with rasterio.open(
        image_path
    ) as src:

        image = src.read()

        transform = src.transform

        crs = src.crs

        width = src.width

        height = src.height

        profile = src.profile.copy()

        print(
            "Original size:",
            width,
            "x",
            height
        )

        print(
            "Bands:",
            src.count
        )

        print(
            "CRS:",
            crs
        )

    # ------------------------------------------------------------
    # Need at least RGB
    # ------------------------------------------------------------

    if image.shape[0] < 3:

        raise ValueError(
            "Satellite image must have "
            "at least 3 bands."
        )

    # ------------------------------------------------------------
    # Take first 3 bands
    #
    # CHW -> HWC
    # ------------------------------------------------------------

    image = image[:3]

    image = np.transpose(
        image,
        (1, 2, 0)
    )

    # ------------------------------------------------------------
    # Convert float
    # ------------------------------------------------------------

    image = image.astype(
        np.float32
    )

    # ------------------------------------------------------------
    # Normalize image
    # ------------------------------------------------------------

    min_value = np.nanmin(
        image
    )

    max_value = np.nanmax(
        image
    )

    print(
        "Minimum:",
        min_value
    )

    print(
        "Maximum:",
        max_value
    )

    if max_value > min_value:

        image = (

            image - min_value

        ) / (

            max_value - min_value

        )

    else:

        image[:] = 0

    # ------------------------------------------------------------
    # Replace invalid values
    # ------------------------------------------------------------

    image = np.nan_to_num(

        image,

        nan=0.0,

        posinf=1.0,

        neginf=0.0
    )

    # ------------------------------------------------------------
    # Resize image
    # ------------------------------------------------------------

    image = cv2.resize(

        image,

        IMAGE_SIZE,

        interpolation=cv2.INTER_LINEAR
    )

    print(
        "Processed image:",
        image.shape
    )

    return (

        image,

        transform,

        crs,

        width,

        height,

        profile
    )


# ================================================================
# STEP 2
# CREATE MASK FROM SHAPEFILE
# ================================================================

def create_mask_from_shapefile(

    shapefile_path,

    transform,

    crs,

    width,

    height
):

    print()
    print("================================================")
    print("CREATING MASK FROM SHAPEFILE")
    print("================================================")

    print(
        "Shapefile:",
        shapefile_path
    )

    if not os.path.exists(
        shapefile_path
    ):

        raise FileNotFoundError(

            f"Shapefile not found: "
            f"{shapefile_path}"
        )

    # ------------------------------------------------------------
    # Read shapefile
    # ------------------------------------------------------------

    gdf = gpd.read_file(
        shapefile_path
    )

    print(
        "Number of polygons:",
        len(gdf)
    )

    print(
        "Shapefile CRS:",
        gdf.crs
    )

    print(
        "Columns:",
        list(gdf.columns)
    )

    # ------------------------------------------------------------
    # Check class_id
    # ------------------------------------------------------------

    if "class_id" not in gdf.columns:

        raise ValueError(

            "The shapefile does not contain "
            "'class_id' field."
        )

    # ------------------------------------------------------------
    # Check CRS
    # ------------------------------------------------------------

    if crs is None:

        raise ValueError(

            "Satellite image does not have CRS."
        )

    if gdf.crs is None:

        raise ValueError(

            "Shapefile does not have CRS."
        )

    # ------------------------------------------------------------
    # Reproject shapefile
    # ------------------------------------------------------------

    if gdf.crs != crs:

        print(
            "CRS mismatch detected."
        )

        print(
            "Reprojecting shapefile..."
        )

        gdf = gdf.to_crs(
            crs
        )

    # ------------------------------------------------------------
    # Prepare polygons
    # ------------------------------------------------------------

    shapes = []

    for index, row in gdf.iterrows():

        geometry = row.geometry

        if geometry is None:

            continue

        if geometry.is_empty:

            continue

        class_id = row[
            "class_id"
        ]

        if pd.isna(
            class_id
        ):

            continue

        try:

            class_id = int(
                class_id
            )

        except Exception:

            print(
                f"Invalid class_id at "
                f"polygon {index}"
            )

            continue

        # --------------------------------------------------------
        # Validate class
        # --------------------------------------------------------

        if class_id not in CLASS_NAMES:

            print(

                f"Invalid class_id "
                f"{class_id} at polygon "
                f"{index}"
            )

            continue

        print(

            f"Polygon {index}: "
            f"class_id={class_id} "
            f"({CLASS_NAMES[class_id]})"
        )

        shapes.append(

            (
                geometry,
                class_id
            )
        )

    # ------------------------------------------------------------
    # Rasterize
    # ------------------------------------------------------------

    print()
    print(
        "Rasterizing polygons..."
    )

    mask = features.rasterize(

        shapes,

        out_shape=(

            height,

            width
        ),

        transform=transform,

        fill=0,

        dtype=np.uint8,

        all_touched=True
    )

    # ------------------------------------------------------------
    # Resize mask
    #
    # VERY IMPORTANT:
    #
    # NEVER use linear interpolation
    # for class masks.
    #
    # Use nearest-neighbor.
    # ------------------------------------------------------------

    mask = cv2.resize(

        mask,

        IMAGE_SIZE,

        interpolation=cv2.INTER_NEAREST
    )

    print(
        "Mask shape:",
        mask.shape
    )

    print(
        "Classes in mask:",
        np.unique(mask)
    )

    # ------------------------------------------------------------
    # Pixel statistics
    # ------------------------------------------------------------

    print()
    print(
        "Mask class statistics:"
    )

    total_pixels = mask.size

    for class_id in range(
        NUM_CLASSES
    ):

        count = np.sum(
            mask == class_id
        )

        percentage = (

            count /
            total_pixels
        ) * 100.0

        print(

            f"  {class_id} - "
            f"{CLASS_NAMES[class_id]:12s} "
            f": {count:10d} pixels "
            f"({percentage:.2f}%)"
        )

    return mask


# ================================================================
# STEP 3
# PYTORCH DATASET
# ================================================================

class SingleImageDataset(
    Dataset
):

    def __init__(
        self,
        image,
        mask
    ):

        self.image = image

        self.mask = mask

    def __len__(
        self
    ):

        return 1

    def __getitem__(
        self,
        index
    ):

        # --------------------------------------------------------
        # HWC -> CHW
        # --------------------------------------------------------

        image = np.transpose(

            self.image,

            (2, 0, 1)
        )

        image = torch.from_numpy(
            image
        ).float()

        mask = torch.from_numpy(
            self.mask
        ).long()

        return (
            image,
            mask
        )


# ================================================================
# U-NET DOUBLE CONV
# ================================================================

class DoubleConv(
    nn.Module
):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(

                in_channels,

                out_channels,

                kernel_size=3,

                padding=1,

                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(

                out_channels,

                out_channels,

                kernel_size=3,

                padding=1,

                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            )
        )

    def forward(
        self,
        x
    ):

        return self.block(
            x
        )


# ================================================================
# U-NET
# ================================================================

class UNet(
    nn.Module
):

    def __init__(
        self,
        num_classes=5
    ):

        super().__init__()

        # ========================================================
        # ENCODER
        # ========================================================

        self.enc1 = DoubleConv(
            3,
            32
        )

        self.enc2 = DoubleConv(
            32,
            64
        )

        self.enc3 = DoubleConv(
            64,
            128
        )

        self.enc4 = DoubleConv(
            128,
            256
        )

        # ========================================================
        # BOTTLENECK
        # ========================================================

        self.bottleneck = DoubleConv(
            256,
            512
        )

        # ========================================================
        # POOLING
        # ========================================================

        self.pool = nn.MaxPool2d(
            kernel_size=2
        )

        # ========================================================
        # DECODER
        # ========================================================

        self.up4 = nn.ConvTranspose2d(

            512,

            256,

            kernel_size=2,

            stride=2
        )

        self.dec4 = DoubleConv(

            512,

            256
        )

        # --------------------------------------------------------

        self.up3 = nn.ConvTranspose2d(

            256,

            128,

            kernel_size=2,

            stride=2
        )

        self.dec3 = DoubleConv(

            256,

            128
        )

        # --------------------------------------------------------

        self.up2 = nn.ConvTranspose2d(

            128,

            64,

            kernel_size=2,

            stride=2
        )

        self.dec2 = DoubleConv(

            128,

            64
        )

        # --------------------------------------------------------

        self.up1 = nn.ConvTranspose2d(

            64,

            32,

            kernel_size=2,

            stride=2
        )

        self.dec1 = DoubleConv(

            64,

            32
        )

        # ========================================================
        # OUTPUT
        # ========================================================

        self.output = nn.Conv2d(

            32,

            num_classes,

            kernel_size=1
        )

    def forward(
        self,
        x
    ):

        # ========================================================
        # ENCODER
        # ========================================================

        e1 = self.enc1(
            x
        )

        p1 = self.pool(
            e1
        )

        e2 = self.enc2(
            p1
        )

        p2 = self.pool(
            e2
        )

        e3 = self.enc3(
            p2
        )

        p3 = self.pool(
            e3
        )

        e4 = self.enc4(
            p3
        )

        p4 = self.pool(
            e4
        )

        # ========================================================
        # BOTTLENECK
        # ========================================================

        b = self.bottleneck(
            p4
        )

        # ========================================================
        # DECODER
        # ========================================================

        d4 = self.up4(
            b
        )

        d4 = torch.cat(

            [
                d4,
                e4
            ],

            dim=1
        )

        d4 = self.dec4(
            d4
        )

        # --------------------------------------------------------

        d3 = self.up3(
            d4
        )

        d3 = torch.cat(

            [
                d3,
                e3
            ],

            dim=1
        )

        d3 = self.dec3(
            d3
        )

        # --------------------------------------------------------

        d2 = self.up2(
            d3
        )

        d2 = torch.cat(

            [
                d2,
                e2
            ],

            dim=1
        )

        d2 = self.dec2(
            d2
        )

        # --------------------------------------------------------

        d1 = self.up1(
            d2
        )

        d1 = torch.cat(

            [
                d1,
                e1
            ],

            dim=1
        )

        d1 = self.dec1(
            d1
        )

        # ========================================================
        # OUTPUT
        # ========================================================

        return self.output(
            d1
        )


# ================================================================
# DICE LOSS
# ================================================================

def dice_loss(
    logits,
    targets
):

    probabilities = torch.softmax(

        logits,

        dim=1
    )

    targets_one_hot = F.one_hot(

        targets,

        num_classes=NUM_CLASSES
    )

    targets_one_hot = (
        targets_one_hot
        .permute(0, 3, 1, 2)
        .float()
    )

    # ------------------------------------------------------------
    # Flatten
    # ------------------------------------------------------------

    probabilities = probabilities.contiguous().view(

        probabilities.shape[0],

        probabilities.shape[1],

        -1
    )

    targets_one_hot = targets_one_hot.contiguous().view(

        targets_one_hot.shape[0],

        targets_one_hot.shape[1],

        -1
    )

    # ------------------------------------------------------------
    # Intersection
    # ------------------------------------------------------------

    intersection = (

        probabilities *
        targets_one_hot

    ).sum(
        dim=2
    )

    denominator = (

        probabilities.sum(
            dim=2
        )

        +

        targets_one_hot.sum(
            dim=2
        )
    )

    dice = (

        2.0 * intersection + 1e-6

    ) / (

        denominator + 1e-6
    )

    return (
        1.0 - dice.mean()
    )


# ================================================================
# TOTAL LOSS
# ================================================================

def segmentation_loss(
    logits,
    targets
):

    cross_entropy = F.cross_entropy(

        logits,

        targets
    )

    dice = dice_loss(

        logits,

        targets
    )

    return (
        cross_entropy +
        dice
    )


# ================================================================
# IOU
# ================================================================

def calculate_iou(
    prediction,
    target
):

    print()
    print(
        "================================================"
    )

    print(
        "IOU / DICE"
    )

    print(
        "================================================"
    )

    prediction = prediction.flatten()

    target = target.flatten()

    all_ious = []

    all_dices = []

    for class_id in range(
        NUM_CLASSES
    ):

        pred_class = (
            prediction == class_id
        )

        target_class = (
            target == class_id
        )

        intersection = np.logical_and(

            pred_class,

            target_class
        ).sum()

        union = np.logical_or(

            pred_class,

            target_class
        ).sum()

        pred_pixels = (
            pred_class.sum()
        )

        target_pixels = (
            target_class.sum()
        )

        # --------------------------------------------------------
        # IoU
        # --------------------------------------------------------

        if union == 0:

            iou = 1.0

        else:

            iou = (

                intersection /
                union
            )

        # --------------------------------------------------------
        # Dice
        # --------------------------------------------------------

        denominator = (

            pred_pixels +
            target_pixels
        )

        if denominator == 0:

            dice = 1.0

        else:

            dice = (

                2.0 *
                intersection /
                denominator
            )

        all_ious.append(
            iou
        )

        all_dices.append(
            dice
        )

        print(

            f"{class_id} - "
            f"{CLASS_NAMES[class_id]:12s} "
            f"| IoU = {iou:.4f} "
            f"| Dice = {dice:.4f}"
        )

    print()
    print(
        "Mean IoU:",
        np.mean(all_ious)
    )

    print(
        "Mean Dice:",
        np.mean(all_dices)
    )


# ================================================================
# SAVE COLORED MASK
# ================================================================

def save_colored_mask(
    mask,
    output_path
):

    height, width = mask.shape

    colored = np.zeros(

        (
            height,
            width,
            3
        ),

        dtype=np.uint8
    )

    for class_id, color in CLASS_COLORS.items():

        colored[
            mask == class_id
        ] = color

    # ------------------------------------------------------------
    # RGB -> BGR
    # ------------------------------------------------------------

    colored = cv2.cvtColor(

        colored,

        cv2.COLOR_RGB2BGR
    )

    cv2.imwrite(

        output_path,

        colored
    )

    print(
        "Saved:",
        output_path
    )


# ================================================================
# SAVE CLASS LEGEND
# ================================================================

def save_legend():

    height = 250

    width = 600

    legend = np.zeros(

        (
            height,
            width,
            3
        ),

        dtype=np.uint8
    )

    for index, class_id in enumerate(
        CLASS_NAMES
    ):

        y = 35 + (
            index * 45
        )

        color = CLASS_COLORS[
            class_id
        ]

        cv2.rectangle(

            legend,

            (
                20,
                y - 20
            ),

            (
                60,
                y + 15
            ),

            color,

            -1
        )

        cv2.putText(

            legend,

            (
                f"{class_id} - "
                f"{CLASS_NAMES[class_id]}"
            ),

            (
                80,
                y
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (
                255,
                255,
                255
            ),

            2
        )

    cv2.imwrite(

        os.path.join(
            OUTPUT_DIR,
            "legend.png"
        ),

        legend
    )


# ================================================================
# MAIN
# ================================================================

def main():

    print()
    print(
        "================================================"
    )

    print(
        "SINGLE IMAGE U-NET"
    )

    print(
        "SEMANTIC SEGMENTATION"
    )

    print(
        "================================================"
    )

    print()
    print(
        "Classes:"
    )

    for class_id, name in CLASS_NAMES.items():

        print(
            f"  {class_id} = {name}"
        )

    # ============================================================
    # STEP 1
    # LOAD IMAGE
    # ============================================================

    (

        image,

        transform,

        crs,

        width,

        height,

        profile

    ) = load_satellite_image(

        IMAGE_PATH
    )

    # ============================================================
    # STEP 2
    # CREATE MASK
    # ============================================================

    mask = create_mask_from_shapefile(

        SHAPEFILE_PATH,

        transform,

        crs,

        width,

        height
    )

    # ============================================================
    # SAVE GROUND TRUTH MASK
    # ============================================================

    save_colored_mask(

        mask,

        os.path.join(

            OUTPUT_DIR,

            "ground_truth.png"
        )
    )

    save_legend()

    # ============================================================
    # STEP 3
    # DATASET
    # ============================================================

    dataset = SingleImageDataset(

        image,

        mask
    )

    loader = DataLoader(

        dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=0
    )

    # ============================================================
    # STEP 4
    # MODEL
    # ============================================================

    print()
    print(
        "================================================"
    )

    print(
        "CREATING U-NET"
    )

    print(
        "================================================"
    )

    model = UNet(

        num_classes=NUM_CLASSES
    )

    model = model.to(
        DEVICE
    )

    # ============================================================
    # OPTIMIZER
    # ============================================================

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=LEARNING_RATE
    )

    # ============================================================
    # STEP 5
    # TRAIN
    # ============================================================

    print()
    print(
        "================================================"
    )

    print(
        "TRAINING"
    )

    print(
        "================================================"
    )

    best_loss = float(
        "inf"
    )

    for epoch in range(
        EPOCHS
    ):

        model.train()

        total_loss = 0.0

        for images, masks in loader:

            images = images.to(
                DEVICE
            )

            masks = masks.to(
                DEVICE
            )

            # ----------------------------------------------------
            # Forward
            # ----------------------------------------------------

            outputs = model(
                images
            )

            # ----------------------------------------------------
            # Loss
            # ----------------------------------------------------

            loss = segmentation_loss(

                outputs,

                masks
            )

            # ----------------------------------------------------
            # Backpropagation
            # ----------------------------------------------------

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
            )

        epoch_loss = (
            total_loss /
            len(loader)
        )

        print(

            f"Epoch "
            f"{epoch + 1:03d}/"
            f"{EPOCHS} "
            f"| Loss: "
            f"{epoch_loss:.6f}"
        )

        # --------------------------------------------------------
        # Save best model
        # --------------------------------------------------------

        if epoch_loss < best_loss:

            best_loss = epoch_loss

            torch.save(

                model.state_dict(),

                MODEL_PATH
            )

    # ============================================================
    # STEP 6
    # LOAD BEST MODEL
    # ============================================================

    print()
    print(
        "================================================"
    )

    print(
        "LOADING BEST MODEL"
    )

    print(
        "================================================"
    )

    model.load_state_dict(

        torch.load(

            MODEL_PATH,

            map_location=DEVICE
        )
    )

    model.eval()

    # ============================================================
    # STEP 7
    # PREDICTION
    # ============================================================

    print()
    print(
        "================================================"
    )

    print(
        "PREDICTION"
    )

    print(
        "================================================"
    )

    with torch.no_grad():

        input_tensor = torch.from_numpy(

            np.transpose(

                image,

                (2, 0, 1)
            )
        ).float()

        input_tensor = (
            input_tensor
            .unsqueeze(0)
            .to(DEVICE)
        )

        output = model(

            input_tensor
        )

        prediction = torch.argmax(

            output,

            dim=1
        )

        prediction = (
            prediction
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

    # ============================================================
    # PRINT PREDICTED CLASSES
    # ============================================================

    print(
        "Predicted classes:",
        np.unique(prediction)
    )

    # ============================================================
    # STEP 8
    # SAVE PREDICTION
    # ============================================================

    save_colored_mask(

        prediction,

        os.path.join(

            OUTPUT_DIR,

            "prediction.png"
        )
    )

    # ============================================================
    # STEP 9
    # SAVE ORIGINAL IMAGE
    # ============================================================

    image_uint8 = (

        image * 255

    ).clip(

        0,

        255

    ).astype(
        np.uint8
    )

    image_bgr = cv2.cvtColor(

        image_uint8,

        cv2.COLOR_RGB2BGR
    )

    cv2.imwrite(

        os.path.join(

            OUTPUT_DIR,

            "original.png"
        ),

        image_bgr
    )

    # ============================================================
    # STEP 10
    # METRICS
    # ============================================================

    calculate_iou(

        prediction,

        mask
    )

    # ============================================================
    # FINISHED
    # ============================================================

    print()
    print(
        "================================================"
    )

    print(
        "COMPLETED"
    )

    print(
        "================================================"
    )

    print()
    print(
        "Model:"
    )

    print(
        MODEL_PATH
    )

    print()
    print(
        "Outputs:"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print(
        "Files generated:"
    )

    print(
        "  original.png"
    )

    print(
        "  ground_truth.png"
    )

    print(
        "  prediction.png"
    )

    print(
        "  legend.png"
    )

    print()
    print(
        "================================================"
    )


# ================================================================
# RUN
# ================================================================

if __name__ == "__main__":

    main()