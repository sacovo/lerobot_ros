from io import BytesIO

import cv2
import numpy as np
import torch
from PIL import Image as PILImage
from sensor_msgs.msg import CompressedImage, Image

from .base import BaseTopic, prefix_names


def ros_image_to_pil(ros_image):
    """
    Convert a ROS Image message to a PIL Image, handling row padding.
    """
    height = ros_image.height
    width = ros_image.width
    encoding = ros_image.encoding
    is_bigendian = ros_image.is_bigendian
    step = ros_image.step
    data = ros_image.data

    # Convert bytes to numpy array
    if isinstance(data, (list, tuple)):
        data = bytes(data)

    # Handle different encodings
    if encoding == "rgb8":
        # RGB 8-bit with potential padding
        channels = 3
        bytes_per_pixel = channels * 1  # 1 byte per channel

        # Reshape considering the step (bytes per row)
        img_array = np.frombuffer(data, dtype=np.uint8).reshape((height, step))

        # Extract only the actual image data (remove padding)
        img_array = img_array[:, : width * bytes_per_pixel].reshape(
            (height, width, channels)
        )
        pil_image = PILImage.fromarray(img_array, "RGB")

    elif encoding == "bgr8":
        # BGR 8-bit with potential padding
        channels = 3
        bytes_per_pixel = channels * 1

        img_array = np.frombuffer(data, dtype=np.uint8).reshape((height, step))
        img_array = img_array[:, : width * bytes_per_pixel].reshape(
            (height, width, channels)
        )
        img_array = img_array[:, :, ::-1]  # BGR to RGB
        pil_image = PILImage.fromarray(img_array, "RGB")

    elif encoding == "rgba8":
        # RGBA 8-bit with potential padding
        channels = 4
        bytes_per_pixel = channels * 1

        img_array = np.frombuffer(data, dtype=np.uint8).reshape((height, step))
        img_array = img_array[:, : width * bytes_per_pixel].reshape(
            (height, width, channels)
        )
        pil_image = PILImage.fromarray(img_array, "RGBA")

    elif encoding == "bgra8":
        # BGRA 8-bit with potential padding
        channels = 4
        bytes_per_pixel = channels * 1

        img_array = np.frombuffer(data, dtype=np.uint8).reshape((height, step))
        img_array = img_array[:, : width * bytes_per_pixel].reshape(
            (height, width, channels)
        )
        img_array = img_array[:, :, [2, 1, 0, 3]]  # BGRA to RGBA
        pil_image = PILImage.fromarray(img_array, "RGBA")

    elif encoding == "mono8":
        # Grayscale 8-bit with potential padding
        bytes_per_pixel = 1

        img_array = np.frombuffer(data, dtype=np.uint8).reshape((height, step))
        img_array = img_array[:, : width * bytes_per_pixel].reshape((height, width))
        pil_image = PILImage.fromarray(img_array, "L")

    elif encoding == "mono16":
        # Grayscale 16-bit with potential padding
        dtype = ">u2" if is_bigendian else "<u2"
        bytes_per_pixel = 2

        # For 16-bit data, step should be divided by 2 since we're reading 2-byte values
        img_array = np.frombuffer(data, dtype=dtype).reshape((height, step // 2))
        img_array = img_array[:, :width].reshape((height, width))
        img_array = (img_array / 256).astype(np.uint8)
        pil_image = PILImage.fromarray(img_array, "L")

    else:
        raise ValueError(f"Unsupported encoding: {encoding}")

    return pil_image


def ros_image_to_numpy(ros_image):
    """
    Convert a ROS Image message directly to an RGB numpy array.
    """
    height = ros_image.height
    width = ros_image.width
    encoding = ros_image.encoding
    step = ros_image.step
    data = ros_image.data

    if isinstance(data, (list, tuple)):
        data = bytes(data)

    if encoding == "rgb8":
        # RGB 8-bit - already in correct format
        img_array = np.frombuffer(data, dtype=np.uint8).reshape((height, step))
        rgb_array = img_array[:, : width * 3].reshape((height, width, 3))
        pil_image = PILImage.fromarray(rgb_array, "RGB")

    elif encoding == "bgr8":
        # BGR 8-bit - convert to RGB
        img_array = np.frombuffer(data, dtype=np.uint8).reshape((height, step))
        bgr_array = img_array[:, : width * 3].reshape((height, width, 3))
        rgb_array = bgr_array[:, :, ::-1]  # BGR to RGB
        pil_image = PILImage.fromarray(rgb_array, "RGB")

    elif encoding == "mono8":
        # Grayscale - convert to RGB by repeating channels
        img_array = np.frombuffer(data, dtype=np.uint8).reshape((height, step))
        gray_array = img_array[:, :width].reshape((height, width))
        rgb_array = np.stack([gray_array, gray_array, gray_array], axis=2)

        pil_image = PILImage.fromarray(rgb_array, "RGB")
    else:
        # For other encodings, use the PIL conversion then convert to array
        pil_image = ros_image_to_pil(ros_image)
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

    return pil_image


class ImageTopic(BaseTopic):
    def __init__(self, height, width, channels, rotate=0, key="main", **kwargs):
        super().__init__(**kwargs)
        self.key = key
        self.height = height
        self.width = width
        self.channels = channels
        self.rotate = rotate

    @staticmethod
    def msg_type():
        return Image

    def is_image(self):
        return True

    def feature_description(self):
        return {
            "dtype": "video",
            "shape": (self.height, self.width, self.channels),  # HxWxC for RGB images
            "names": prefix_names(["height", "width", "channels"], self.topic_name),
        }

    def to_tensor(self, msg: Image) -> torch.Tensor:
        """Convert a ROS Image message to a PyTorch tensor."""
        pil_image = ros_image_to_numpy(msg)

        if self.rotate:
            pil_image = pil_image.rotate(self.rotate, expand=True)
        if pil_image.size != (self.width, self.height):
            img = np.array(pil_image)
            img = cv2.resize(
                img, (self.width, self.height), interpolation=cv2.INTER_NEAREST
            )
        else:
            img = np.array(pil_image)
        # Resize if necessary
        return torch.tensor(
            img,
            dtype=torch.uint8,
        )

    def from_tensor(self, tensor: torch.Tensor) -> Image:
        """Convert a PyTorch tensor to a ROS Image message."""

        img_array = tensor.numpy()
        img = self.bridge.cv2_to_imgmsg(img_array, encoding="rgb8")

        return img


class ImageCompressedTopic(ImageTopic):
    @staticmethod
    def msg_type():
        return CompressedImage

    def feature_description(self):
        return super().feature_description()

    def to_tensor(self, msg: CompressedImage) -> torch.Tensor:
        """Convert a ROS compressed Image message to a PyTorch tensor."""
        # Load image from compressed data using PIL
        pil_image = PILImage.open(BytesIO(msg.data))
        # Resize if not matching expected size

        if self.rotate:
            pil_image = pil_image.rotate(self.rotate, expand=True)

        if pil_image.size != (self.width, self.height):
            pil_image = pil_image.resize((self.width, self.height))

        # Convert to numpy array
        img_array = np.asarray(pil_image)
        return torch.tensor(img_array, dtype=torch.uint8)

    def from_tensor(self, tensor: torch.Tensor) -> CompressedImage:
        """Convert a PyTorch tensor to a ROS compressed Image message."""
        return super().from_tensor(tensor)
