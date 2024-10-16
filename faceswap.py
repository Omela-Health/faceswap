import cv2
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import numpy as np
import argparse
from gfpgan import GFPGANer
import os
import requests
from datetime import datetime


class FaceSwapper:

    def download_models(self):
        """Download the models for face swap and face enhancement"""

        def download_file(url, filename):
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(filename, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

        # Download the inswapper model
        if not os.path.exists("inswapper_128.onnx"):
            print("Downloading inswapper model...")
            download_file(
                "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
                "inswapper_128.onnx",
            )

        # Create GFPGAN weights directory
        os.makedirs("./gfpgan/weights", exist_ok=True)

        # Download GFPGAN models
        models = {
            "realesr-general-x4v3.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
            "GFPGANv1.4.pth": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
            "RestoreFormer.pth": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth",
        }

        for model_name, url in models.items():
            file_path = f"./gfpgan/weights/{model_name}"
            if not os.path.exists(file_path):
                print(f"Downloading {model_name}...")
                download_file(url, file_path)

        # Initialize the FaceAnalysis module
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0, det_size=(512, 512))

        # Load the face swapper
        self.swapper = insightface.model_zoo.get_model(
            "inswapper_128.onnx", download=False, download_zip=False
        )

        self.gfpgan = GFPGANer(
            model_path="./gfpgan/weights/GFPGANv1.4.pth",
            upscale=1,  # 4 is too much
            arch="clean",
            channel_multiplier=2,
        )

    def face_swap(self, source_img, target_img):

        # Ensure models are downloaded before proceeding
        if (
            not hasattr(self, "app")
            or not hasattr(self, "swapper")
            or not hasattr(self, "gfpgan")
        ):
            raise RuntimeError(
                "Models not downloaded. Please run the download_models() method before proceeding."
            )

        # Analyze faces in both images
        source_faces = self.app.get(source_img)
        target_faces = self.app.get(target_img)

        # If no faces found in either image, return the original target image
        if len(source_faces) == 0 or len(target_faces) == 0:
            return target_img

        # Use the first detected face from each image
        source_face = source_faces[0]

        # Create a copy of the target image to modify
        result_img = target_img.copy()

        # Swap faces for all detected faces in the target image
        for target_face in target_faces:
            result_img = self.swapper.get(
                result_img, target_face, source_face, paste_back=True
            )

        return result_img

    def enhance_image(self, img):
        _, _, output = self.gfpgan.enhance(
            img, has_aligned=False, only_center_face=False, paste_back=True
        )
        return output

    def process_images(self, source_path, target_path, enhance=True):

        # Print start time
        start_time = datetime.now()
        print(f"Setup started at: {start_time}")

        # Load source and target images
        if isinstance(source_path, str):
            source_img = cv2.imread(source_path)
        else:
            # Assuming source_path is a PIL Image
            source_img = cv2.cvtColor(np.array(source_path), cv2.COLOR_RGB2BGR)

        if isinstance(target_path, str):
            target_img = cv2.imread(target_path)
        else:
            # Assuming target_path is a PIL Image
            target_img = cv2.cvtColor(np.array(target_path), cv2.COLOR_RGB2BGR)

        if source_img is None or target_img is None:
            print("Error: Failed to load source or target image.")
            return

        # Perform face swap
        result = self.face_swap(source_img, target_img)

        if result is None:
            print(
                "Error: Face swap operation failed. No faces detected or swap unsuccessful."
            )
            return

        elapsed_time = datetime.now() - start_time
        print(f"Swap took {elapsed_time}")

        # Enhance the result
        if enhance:
            result = self.enhance_image(result)

            elapsed_time = datetime.now() - start_time
            print(f"Enhance took {elapsed_time}")

        # Print elapsed time
        elapsed_time = datetime.now() - start_time
        print(f"Face swap process completed in {elapsed_time}")

        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face swap application")
    parser.add_argument(
        "-source", type=str, required=True, help="Path to the source image"
    )
    parser.add_argument(
        "-target", type=str, required=True, help="Path to the target image"
    )
    args = parser.parse_args()

    face_swapper = FaceSwapper()
    face_swapper.download_models()

    img = face_swapper.process_images(args.source, args.target, enhance=True)

    cv2.imwrite("result.jpg", img)
