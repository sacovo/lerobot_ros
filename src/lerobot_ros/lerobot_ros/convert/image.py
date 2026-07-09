import cv2
import numpy as np
import torch
from sensor_msgs.msg import CompressedImage, Image

from .base import BaseTopic, prefix_names


def ros_image_to_numpy(ros_image):
    """
    Convert a ROS Image message directly to an RGB numpy array.
    """
    height = ros_image.height
    width = ros_image.width
    encoding = ros_image.encoding
    is_bigendian = ros_image.is_bigendian
    step = ros_image.step
    data = ros_image.data

    if isinstance(data, (list, tuple)):
        data = bytes(data)

    if encoding == "rgb8":
        # RGB 8-bit
        img_array = np.frombuffer(data, dtype=np.uint8).reshape((height, step))
        return img_array[:, : width * 3].reshape((height, width, 3))

    elif encoding == "bgr8":
        # BGR 8-bit
        img_array = np.frombuffer(data, dtype=np.uint8).reshape((height, step))
        bgr_array = img_array[:, : width * 3].reshape((height, width, 3))
        return cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)

    elif encoding == "mono8":
        # Grayscale - convert to RGB by repeating channels
        img_array = np.frombuffer(data, dtype=np.uint8).reshape((height, step))
        gray_array = img_array[:, :width].reshape((height, width))
        return cv2.cvtColor(gray_array, cv2.COLOR_GRAY2RGB)

    elif encoding == "rgba8":
        # RGBA 8-bit with potential padding; drop the alpha channel
        img_array = np.frombuffer(data, dtype=np.uint8).reshape((height, step))
        img_array = img_array[:, : width * 4].reshape((height, width, 4))
        return np.ascontiguousarray(img_array[:, :, :3])

    elif encoding == "bgra8":
        # BGRA 8-bit with potential padding; drop alpha and reorder to RGB
        img_array = np.frombuffer(data, dtype=np.uint8).reshape((height, step))
        img_array = img_array[:, : width * 4].reshape((height, width, 4))
        return np.ascontiguousarray(img_array[:, :, [2, 1, 0]])

    elif encoding == "mono16":
        # Grayscale 16-bit with potential padding, downshifted to 8-bit then
        # replicated to RGB (same as the mono8 path above)
        dtype = ">u2" if is_bigendian else "<u2"
        # For 16-bit data, step should be divided by 2 since we're reading 2-byte values
        img_array = np.frombuffer(data, dtype=dtype).reshape((height, step // 2))
        img_array = img_array[:, :width].reshape((height, width))
        gray_array = (img_array / 256).astype(np.uint8)
        return cv2.cvtColor(gray_array, cv2.COLOR_GRAY2RGB)

    else:
        raise ValueError(f"Unsupported encoding: {encoding}")


def _rotate_expand(img, angle):
    """
    Rotate img (numpy array) by angle degrees, expanding the canvas to fit the
    whole rotated image -- matching PIL's Image.rotate(angle, expand=True)
    (both PIL and cv2.getRotationMatrix2D use counter-clockwise positive angles).
    """
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(round(h * sin + w * cos))
    new_h = int(round(h * cos + w * sin))

    M[0, 2] += (new_w / 2.0) - cx
    M[1, 2] += (new_h / 2.0) - cy

    return cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_NEAREST)


class ImageTopic(BaseTopic):
    def __init__(self, height, width, channels, rotate=0,  **kwargs):
        super().__init__(**kwargs)
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
        img = ros_image_to_numpy(msg)

        if self.rotate:
            rot = self.rotate % 360
            if rot == 90:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif rot == 180:
                img = cv2.rotate(img, cv2.ROTATE_180)
            elif rot == 270:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                img = _rotate_expand(img, self.rotate)

        if img.shape[1] != self.width or img.shape[0] != self.height:
            img = cv2.resize(
                img, (self.width, self.height), interpolation=cv2.INTER_NEAREST
            )

        return torch.tensor(
            img,
            dtype=torch.uint8,
        )

    def from_tensor(self, tensor: torch.Tensor) -> Image:
        """Convert a PyTorch tensor to a ROS Image message."""

        img_array = tensor.numpy()
        img = self.bridge.cv2_to_imgmsg(img_array, encoding="rgb8")

        return img


def _decode_jpeg_nvjpeg(np_arr: np.ndarray) -> np.ndarray:
    """Decode JPEG bytes via NVJPEG (GPU hardware decode, e.g. on Jetson)
    instead of libjpeg-turbo on CPU, returning an RGB (H, W, C) uint8 numpy
    array -- verified bit-identical to the cv2.imdecode + COLOR_BGR2RGB path
    below (mode=RGB forces 3-channel output the same way IMREAD_COLOR does).
    Bringing the result back to CPU immediately keeps the rest of
    to_tensor()'s rotate/resize path (cv2-based) unchanged; only the decode
    itself moves to the GPU.

    Requires a CUDA device and a torchvision build with NVJPEG support.
    Verify with:
        python3 -c "import torch,torchvision; from torchvision.io import \\
        encode_jpeg,decode_jpeg; t=torch.randint(0,255,(3,64,64), \\
        dtype=torch.uint8); out=decode_jpeg(encode_jpeg(t),device='cuda'); \\
        print(out.device)"
    """
    from torchvision.io import ImageReadMode, decode_jpeg

    raw = torch.tensor(np_arr)
    img_tensor = decode_jpeg(raw, mode=ImageReadMode.RGB, device="cuda")
    return img_tensor.permute(1, 2, 0).cpu().numpy()


class ImageCompressedTopic(ImageTopic):
    def __init__(self, use_nvjpeg=False, **kwargs):
        super().__init__(**kwargs)
        self.use_nvjpeg = use_nvjpeg
        if self.use_nvjpeg and not torch.cuda.is_available():
            raise ValueError(
                f"Topic '{kwargs.get('topic_name')}' has use_nvjpeg=true but no CUDA "
                "device is available -- fix at startup rather than failing per-frame."
            )

    @staticmethod
    def msg_type():
        return CompressedImage

    def feature_description(self):
        return super().feature_description()

    def to_tensor(self, msg: CompressedImage) -> torch.Tensor:
        """Convert a ROS compressed Image message to a PyTorch tensor."""
        data = msg.data
        if isinstance(data, (list, tuple)):
            data = bytes(data)

        np_arr = np.frombuffer(data, dtype=np.uint8)

        if self.use_nvjpeg:
            img = _decode_jpeg_nvjpeg(np_arr)
        else:
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode compressed image")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.rotate:
            rot = self.rotate % 360
            if rot == 90:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif rot == 180:
                img = cv2.rotate(img, cv2.ROTATE_180)
            elif rot == 270:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                img = _rotate_expand(img, self.rotate)

        if img.shape[1] != self.width or img.shape[0] != self.height:
            img = cv2.resize(
                img, (self.width, self.height), interpolation=cv2.INTER_NEAREST
            )

        return torch.tensor(img, dtype=torch.uint8)

    def from_tensor(self, tensor: torch.Tensor) -> CompressedImage:
        """Convert a PyTorch tensor to a ROS compressed Image message."""
        return super().from_tensor(tensor)
