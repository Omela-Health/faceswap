import cv2
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import numpy as np
import argparse
from gfpgan import GFPGANer
import os


class FaceSwapper:

    def __init__(self):
        # Initialize the FaceAnalysis module
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0, det_size=(512, 512))

        # Load the face swapper
        self.swapper = insightface.model_zoo.get_model(
            "inswapper_128.onnx", download=False, download_zip=False
        )

        self.gfpgan = GFPGANer(
            model_path="gfpgan/weights/GFPGANv1.4.pth",
            upscale=4,
            arch="clean",
            channel_multiplier=2,
        )

    def download_models(self):
        """Download the models for face swap and face enhancement"""

        # Download the inswapper model
        # Check if the inswapper model already exists
        if not os.path.exists("inswapper_128.onnx"):
            os.system(
                "wget https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx -P ./"
            )

        # Download the GFPGAN models
        # Check if the GFPGAN models already exist
        if not os.path.exists("gfpgan/weights"):
            os.system("mkdir -p gfpgan/weights")

            # Download the RealESRGAN model
            if not os.path.exists("gfpgan/weights/realesr-general-x4v3.pth"):
                os.system(
                    "wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth -P ./gfpgan/weights"
                )

            # Download the GFPGANv1.4 model
            if not os.path.exists("gfpgan/weights/GFPGANv1.4.pth"):
                os.system(
                    "wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth -P ./gfpgan/weights"
                )

            # Download the RestoreFormer model
            if not os.path.exists("gfpgan/weights/RestoreFormer.pth"):
                os.system(
                    "wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth -P ./gfpgan/weights"
                )

    def face_swap(self, source_img, target_img):
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

    def process_images(self, source_path, target_path):
        # Load source and target images
        source_img = cv2.imread(source_path)
        target_img = cv2.imread(target_path)

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

        # Save the raw result
        cv2.imwrite("raw_result.jpg", result)
        print("Raw face swap result saved as 'raw_result.jpg'")

        # Enhance the result
        enhanced_result = self.enhance_image("raw_result.jpg")
        cv2.imwrite("enhanced_result.jpg", enhanced_result)
        print("Enhanced face swap result saved as 'enhanced_result.jpg'")

        print("Face swap process completed.")


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
    face_swapper.process_images(args.source, args.target)
