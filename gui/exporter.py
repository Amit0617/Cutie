import os
from typing import List

from PIL import Image
import numpy as np
import cv2
import av


def convert_frames_to_video(
        image_folder: str,
        output_path: str,
        fps: int = 24,
        bitrate: int = 1,  # in Mbps
        progress_callback=None) -> None:
    # support common image extensions (rgba frames are in png)
    images = [img for img in sorted(os.listdir(image_folder))
              if img.lower().endswith((".jpg", ".jpeg", ".png"))]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    output = av.open(output_path, mode="w")

    stream = output.add_stream("h264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.bit_rate = bitrate * (10**7)

    for i, img_path in enumerate(images):
        img_file = os.path.join(image_folder, img_path)
        img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Failed to read image '{img_file}'")

        # If image has alpha and we're not preserving alpha, composite over green background (same as GUI)
        if img.ndim == 3 and img.shape[2] == 4:
            alpha = img[:, :, 3:4].astype(np.float32) / 255.0
            rgb = img[:, :, :3].astype(np.float32)
            green = np.array([0, 255, 0], dtype=np.float32)
            img = (rgb * alpha + green[np.newaxis, np.newaxis, :] * (1 - alpha)).astype(np.uint8)
        elif img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        frame = av.VideoFrame.from_ndarray(img, format='bgr24')
        packet = stream.encode(frame)
        output.mux(packet)

        if progress_callback is not None and i % 10 == 0:
            progress_callback(i / len(images))

    # flush
    packet = stream.encode(None)
    output.mux(packet)

    output.close()


def convert_mask_to_binary(mask_folder: str,
                           output_path: str,
                           target_objects: List[int],
                           progress_callback=None) -> None:
    masks = [img for img in sorted(os.listdir(mask_folder)) if img.endswith(".png")]

    for i, mask_path in enumerate(masks):
        mask = Image.open(os.path.join(mask_folder, mask_path))
        mask = np.array(mask)
        mask = np.where(np.isin(mask, target_objects), 255, 0)
        cv2.imwrite(os.path.join(output_path, mask_path), mask)

        if progress_callback is not None and i % 10 == 0:
            progress_callback(i / len(masks))
