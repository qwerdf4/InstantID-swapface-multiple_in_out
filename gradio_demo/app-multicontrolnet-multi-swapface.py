import sys
sys.path.append("./")

from typing import Tuple

import os
import cv2
import math
import torch
import random
import numpy as np
import argparse
import onnxruntime
import PIL
from PIL import Image

import diffusers
from diffusers.utils import load_image
from diffusers.models import ControlNetModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel

from huggingface_hub import hf_hub_download

from insightface.app import FaceAnalysis

from style_template import styles
from pipeline_stable_diffusion_xl_instantid_full_inpaint import StableDiffusionXLInstantIDInpaintPipeline
from model_util import load_models_xl, get_torch_device, torch_gc
from controlnet_util import openpose, get_depth_map, get_canny_image

import gradio as gr


# global variable
MAX_SEED = np.iinfo(np.int32).max
device = get_torch_device()
dtype = torch.float16 if str(device).__contains__("cuda") else torch.float32
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "(No style)"

# Load face encoder
app = FaceAnalysis(
    name="antelopev2",
    root="./",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
app.prepare(ctx_id=0, det_size=(640, 640))

# Path to InstantID models
face_adapter = f"./checkpoints/ip-adapter.bin"
controlnet_path = f"./checkpoints/ControlNetModel"

# Load pipeline face ControlNetModel
controlnet_identitynet = ControlNetModel.from_pretrained(
    controlnet_path, torch_dtype=dtype
)

# controlnet-pose
controlnet_pose_model = "thibaud/controlnet-openpose-sdxl-1.0"
controlnet_canny_model = "diffusers/controlnet-canny-sdxl-1.0"
controlnet_depth_model = "diffusers/controlnet-depth-sdxl-1.0-small"

controlnet_pose = ControlNetModel.from_pretrained(
    controlnet_pose_model, torch_dtype=dtype
).to(device)
controlnet_canny = ControlNetModel.from_pretrained(
    controlnet_canny_model, torch_dtype=dtype
).to(device)
controlnet_depth = ControlNetModel.from_pretrained(
    controlnet_depth_model, torch_dtype=dtype
).to(device)

controlnet_map = {
    "pose": controlnet_pose,
    "canny": controlnet_canny,
    "depth": controlnet_depth,
}
controlnet_map_fn = {
    "pose": openpose,
    "canny": get_canny_image,
    "depth": get_depth_map,
}


def main(pretrained_model_name_or_path="wangqixun/YamerMIX_v8", enable_lcm_arg=False):
    if pretrained_model_name_or_path.endswith(
        ".ckpt"
    ) or pretrained_model_name_or_path.endswith(".safetensors"):
        scheduler_kwargs = hf_hub_download(
            repo_id="wangqixun/YamerMIX_v8",
            subfolder="scheduler",
            filename="scheduler_config.json",
        )

        (tokenizers, text_encoders, unet, _, vae) = load_models_xl(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            scheduler_name=None,
            weight_dtype=dtype,
        )

        scheduler = diffusers.EulerDiscreteScheduler.from_config(scheduler_kwargs)
        pipe = StableDiffusionXLInstantIDInpaintPipeline(
            vae=vae,
            text_encoder=text_encoders[0],
            text_encoder_2=text_encoders[1],
            tokenizer=tokenizers[0],
            tokenizer_2=tokenizers[1],
            unet=unet,
            scheduler=scheduler,
            controlnet=[controlnet_identitynet],
        ).to(device)

    else:
        pipe = StableDiffusionXLInstantIDInpaintPipeline.from_pretrained(
            pretrained_model_name_or_path,
            controlnet=[controlnet_identitynet],
            torch_dtype=dtype,
            safety_checker=None,
            feature_extractor=None,
        ).to(device)

        pipe.scheduler = diffusers.EulerDiscreteScheduler.from_config(
            pipe.scheduler.config
        )

    pipe.load_ip_adapter_instantid(face_adapter)
    # load and disable LCM
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
    pipe.disable_lora()

    def toggle_lcm_ui(value):
        if value:
            return (
                gr.update(minimum=0, maximum=100, step=1, value=10),
                gr.update(minimum=0.1, maximum=20.0, step=0.1, value=1.5),
                gr.update(minimum=0, maximum=1.5, step=0.05, value=0.70),
                gr.update(minimum=0, maximum=1.5, step=0.05, value=0.70),
                gr.update(minimum=0, maximum=1.5, step=0.05, value=1.0),
                gr.update(minimum=0, maximum=1.5, step=0.05, value=0.70),
            )
        else:
            return (
                gr.update(minimum=5, maximum=100, step=1, value=30),
                gr.update(minimum=0.1, maximum=20.0, step=0.1, value=3),
                gr.update(minimum=0, maximum=1.5, step=0.05, value=0.80),
                gr.update(minimum=0, maximum=1.5, step=0.05, value=0.80),
                gr.update(minimum=0, maximum=1.5, step=0.05, value=0.8),
                gr.update(minimum=0, maximum=1.5, step=0.05, value=0.40),
            )

    def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
        if randomize_seed:
            seed = random.randint(0, MAX_SEED)
        return seed

    def remove_tips():
        return gr.update(visible=False)

    def convert_from_image_to_cv2(img: Image) -> np.ndarray:
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    def convert_from_cv2_to_image(img: np.ndarray) -> Image:
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    def draw_kps(
        image_pil,
        kps,
        color_list=[
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
        ],
    ):
        stickwidth = 4
        limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
        kps = np.array(kps)

        w, h = image_pil.size
        out_img = np.zeros([h, w, 3])

        for i in range(len(limbSeq)):
            index = limbSeq[i]
            color = color_list[index[0]]

            x = kps[index][:, 0]
            y = kps[index][:, 1]
            length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
            polygon = cv2.ellipse2Poly(
                (int(np.mean(x)), int(np.mean(y))),
                (int(length / 2), stickwidth),
                int(angle),
                0,
                360,
                1,
            )
            out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
        out_img = (out_img * 0.6).astype(np.uint8)

        for idx_kp, kp in enumerate(kps):
            color = color_list[idx_kp]
            x, y = kp
            out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

        out_img_pil = Image.fromarray(out_img.astype(np.uint8))
        return out_img_pil

    def resize_img(
        input_image,
        max_side=1280,
        min_side=1024,
        size=None,
        pad_to_max_side=False,
        mode=PIL.Image.BILINEAR,
        base_pixel_number=64,
    ):
        w, h = input_image.size
        if size is not None:
            w_resize_new, h_resize_new = size
        else:
            ratio = min_side / min(h, w)
            w, h = round(ratio * w), round(ratio * h)
            ratio = max_side / max(h, w)
            input_image = input_image.resize([round(ratio * w), round(ratio * h)], mode)
            w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
            h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
        input_image = input_image.resize([w_resize_new, h_resize_new], mode)

        if pad_to_max_side:
            res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
            offset_x = (max_side - w_resize_new) // 2
            offset_y = (max_side - h_resize_new) // 2
            res[
                offset_y : offset_y + h_resize_new, offset_x : offset_x + w_resize_new
            ] = np.array(input_image)
            input_image = Image.fromarray(res)
        return input_image

    def apply_style(
        style_name: str, positive: str, negative: str = ""
    ) -> Tuple[str, str]:
        p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
        return p.replace("{prompt}", positive), n + " " + negative

    def get_face_occluder():
        model_path = r"./models/face_occluder.onnx" # Your face_occluder.onnx path
        FACE_OCCLUDER = onnxruntime.InferenceSession(model_path)
        return FACE_OCCLUDER

    def create_occlusion_mask(img_face, kernel_size):
        face_occluder = get_face_occluder()
        prepare_face = cv2.resize(img_face, face_occluder.get_inputs()[0].shape[1:3][::-1])
        prepare_face = np.expand_dims(prepare_face, axis=0).astype(np.float32) / 255
        prepare_face = prepare_face.transpose(0, 1, 2, 3)
        occlusion_mask = face_occluder.run(None,
                                           {
                                               face_occluder.get_inputs()[0].name: prepare_face
                                           })[0][0]
        occlusion_mask = occlusion_mask.transpose(0, 1, 2).clip(0, 1).astype(np.float32)
        occlusion_mask = cv2.resize(occlusion_mask, img_face.shape[:2][::-1])
        occlusion_mask = (cv2.GaussianBlur(occlusion_mask.clip(0, 1), (0, 0), 5).clip(0.5, 1) - 0.5) * 2
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        occlusion_mask = cv2.dilate(occlusion_mask,kernel)
        return occlusion_mask

    def swap_to_gallery(images):
        return gr.update(value=images, visible=True), gr.update(visible=True), gr.update(visible=False)

    def remove_back_to_files():
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

    def generate_image(
        face_images,
        pose_images,
        prompt,
        negative_prompt,
        style_name,
        num_steps,
        identitynet_strength_ratio,
        adapter_strength_ratio,
        pose_strength,
        canny_strength,
        depth_strength,
        controlnet_selection,
        guidance_scale,
        seed,
        scheduler,
        enable_LCM,
        enhance_face_region,
        mask_area,
        denoise_strength,
        progress=gr.Progress(track_tqdm=True),
    ):

        if enable_LCM:
            pipe.scheduler = diffusers.LCMScheduler.from_config(pipe.scheduler.config)
            pipe.enable_lora()
        else:
            pipe.disable_lora()
            scheduler_class_name = scheduler.split("-")[0]

            add_kwargs = {}
            if len(scheduler.split("-")) > 1:
                add_kwargs["use_karras_sigmas"] = True
            if len(scheduler.split("-")) > 2:
                add_kwargs["algorithm_type"] = "sde-dpmsolver++"
            scheduler = getattr(diffusers, scheduler_class_name)
            pipe.scheduler = scheduler.from_config(pipe.scheduler.config, **add_kwargs)

        if face_images is None:
            raise gr.Error(
                f"Cannot find any input face image! Please upload the face image"
            )

        if prompt is None:
            prompt = "a person"

        # apply the style template
        prompt, negative_prompt = apply_style(style_name, prompt, negative_prompt)
        img_results=[]
        face_emb = []
        for i in face_images:
            face_image = load_image(i)
            face_image = resize_img(face_image, max_side=1024)
            face_image_cv2 = convert_from_image_to_cv2(face_image)
            height, width, _ = face_image_cv2.shape
            # Extract face features
            face_info = app.get(face_image_cv2)
            if len(face_info) == 0:
                raise gr.Error(
                    f"Unable to detect a face in the image. Please upload a different photo with a clear face."
                )
            face_info = sorted(
                face_info,
                key=lambda x: (x["bbox"][2] - x["bbox"][0]) * x["bbox"][3] - x["bbox"][1],)[-1]  # only use the maximum face
            face_emb.append(face_info['embedding'])
        face_emb = sum(face_emb)/len(face_emb)
        if pose_images is not None:
            for i in pose_images:
                pose_image_original = load_image(i)
                pose_image = resize_img(pose_image_original, max_side=1024)
                img_controlnet = pose_image
                pose_image_cv2 = convert_from_image_to_cv2(pose_image)
                face_info = app.get(pose_image_cv2)
                if len(face_info) == 0:
                    raise gr.Error(
                        f"Cannot find any face in the reference image! Please upload another person image"
                    )
                face_info = face_info[-1]
                face_kps = draw_kps(pose_image, face_info["kps"])
                width, height = face_kps.size
                occlusion_mask = create_occlusion_mask(pose_image_cv2, mask_area)
                if enhance_face_region:
                    control_mask = np.zeros([height, width, 3])
                    x1, y1, x2, y2 = face_info["bbox"]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    control_mask[y1:y2, x1:x2] = 255
                    control_mask = Image.fromarray(control_mask.astype(np.uint8))
                else:
                    control_mask = None

                if len(controlnet_selection) > 0:
                    controlnet_scales = {
                        "pose": pose_strength,
                        "canny": canny_strength,
                        "depth": depth_strength,
                    }
                    pipe.controlnet = MultiControlNetModel(
                        [controlnet_identitynet]
                        + [controlnet_map[s] for s in controlnet_selection]
                    )
                    control_scales = [float(identitynet_strength_ratio)] + [
                        controlnet_scales[s] for s in controlnet_selection
                    ]
                    control_images = [face_kps] + [
                        controlnet_map_fn[s](img_controlnet).resize((width, height))
                        for s in controlnet_selection
                    ]
                else:
                    pipe.controlnet = controlnet_identitynet
                    control_scales = float(identitynet_strength_ratio)
                    control_images = face_kps

                generator = torch.Generator(device=device).manual_seed(seed)

                print("Start inference...")
                print(f"[Debug] Prompt: {prompt}, \n[Debug] Neg Prompt: {negative_prompt}")

                pipe.set_ip_adapter_scale(adapter_strength_ratio)
                images = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image_embeds=face_emb,
                    image=pose_image,
                    control_image=control_images,
                    mask_image=occlusion_mask,
                    control_mask=control_mask,
                    controlnet_conditioning_scale=control_scales,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    generator=generator,
                    strength=denoise_strength,
                ).images[0]
                img_results.append(images)
        else:
            raise gr.Error(
                f"Unable to detect a face in the pose image. Please upload different photo with clear face."
            )
        return img_results, gr.update(visible=True)

    # Description
    title = r"""
    <h1 align="center">Unofficial InstantID for swap face.</h1>
    """

    description = r"""
    <b>UnOfficial 🤗 Gradio demo</b> for <a href='https://github.com/InstantID/InstantID' target='_blank'><b>InstantID: Zero-shot Identity-Preserving Generation in Seconds</b></a>.<br>

    How to use:<br>
    1. Upload one or more images with face, can be of different people, faces will fusion.
    2. upload one or more images as reference for the face pose, and if multiple images are uploaded, multiple results will be output.
    3. (Optional) You can select multiple ControlNet models to control the generation process. for swap face, canny and depth make facial expressions consistent.
    4. (Optional) Enter a text prompt, as done in normal text-to-image models.
    5. Click the <b>Submit</b> button to begin customization.
    6. Share your customized photo with your friends and enjoy! 😊"""

    tips = r"""
    ### Usage tips of InstantID
    1. If you're not satisfied with the similarity, try increasing the weight of "IdentityNet Strength" and "Adapter Strength."    
    2. If you feel that the saturation is too high, first decrease the Adapter strength. If it remains too high, then decrease the IdentityNet strength.
    3. you can also adjust the denoising strength.
    4. for onle swap face, prompt is not necessary.
    5. adjust the strength of sample steps and Guidance scale to control the generation process, keep the Guidance scale low.
    6. If you find that realistic style is not good enough, go for our Github repo and use a more realistic base model.
    """

    css = """
    .gradio-container {width: 85% !important}
    """
    with gr.Blocks(css=css) as demo:
        # description
        gr.Markdown(title)
        gr.Markdown(description)

        with gr.Row():
            with gr.Column():
                face_files = gr.Files(
                    label="Upload a or more photos of your face", file_types=['image']
                )
                face_uploaded_files = gr.Gallery(label="Your images", visible=False, columns=5, rows=1, height=200)
                with gr.Column(visible=False) as face_clear_button:
                    face_remove_and_reupload = gr.ClearButton(value="Remove and upload new ones", components=face_files,
                                                         size="sm")
                pose_files = gr.Files(
                    label="Upload a or more reference pose images (Optional)",
                    file_types=['image'],
                )
                pose_uploaded_files = gr.Gallery(label="Your images", visible=False, columns=5, rows=1, height=200)
                with gr.Column(visible=False) as pose_clear_button:
                    pose_remove_and_reupload = gr.ClearButton(value="Remove and upload new ones", components=pose_files,
                                                         size="sm")
                # prompt
                prompt = gr.Textbox(
                    label="Prompt",
                    info="Give simple prompt is enough to achieve good face fidelity",
                    placeholder="A photo of a person",
                    value="",
                )

                submit = gr.Button("Submit", variant="primary")
                enable_LCM = gr.Checkbox(
                    label="Enable Fast Inference with LCM", value=enable_lcm_arg,
                    info="LCM speeds up the inference step, the trade-off is the quality of the generated image. It performs better with portrait face images rather than distant faces",
                )

                mask_area = gr.Slider(
                    label="Face Mask Area (If you find that face doesn't fit well, increase it)",
                    minimum=0,
                    maximum=60,
                    step=3,
                    value=0,
                )
                style = gr.Dropdown(
                    label="Style template",
                    choices=STYLE_NAMES,
                    value=DEFAULT_STYLE_NAME,
                )

                # strength
                identitynet_strength_ratio = gr.Slider(
                    label="IdentityNet strength (for fidelity)",
                    minimum=0,
                    maximum=1.5,
                    step=0.05,
                    value=0.8,
                )
                adapter_strength_ratio = gr.Slider(
                    label="Image adapter strength (for detail)",
                    minimum=0,
                    maximum=1.5,
                    step=0.05,
                    value=0.4,
                )
                with gr.Accordion("Controlnet"):
                    controlnet_selection = gr.CheckboxGroup(
                        ["pose", "canny", "depth"], label="Controlnet", value=["canny", "depth"],
                        info="Use pose for skeleton inference, canny for edge detection, and depth for depth map estimation. You can try all three to control the generation process"
                    )
                    pose_strength = gr.Slider(
                        label="Pose strength",
                        minimum=0,
                        maximum=1.5,
                        step=0.05,
                        value=0.40,
                    )
                    canny_strength = gr.Slider(
                        label="Canny strength",
                        minimum=0,
                        maximum=1.5,
                        step=0.05,
                        value=0.80,
                    )
                    depth_strength = gr.Slider(
                        label="Depth strength",
                        minimum=0,
                        maximum=1.5,
                        step=0.05,
                        value=0.80,
                    )
                    denoise_strength = gr.Slider(
                        label="denoising strength",
                        minimum=0,
                        maximum=1.0,
                        step=0.01,
                        value=0.8,
                    )
                with gr.Accordion(open=True, label="Advanced Options"):
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="low quality",
                        value="(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
                    )
                    num_steps = gr.Slider(
                        label="Number of sample steps",
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=5 if enable_lcm_arg else 30,
                    )
                    guidance_scale = gr.Slider(
                        label="Guidance scale",
                        minimum=0.1,
                        maximum=20.0,
                        step=0.1,
                        value=0.0 if enable_lcm_arg else 3.0,
                    )
                    seed = gr.Slider(
                        label="Seed",
                        minimum=0,
                        maximum=MAX_SEED,
                        step=1,
                        value=42,
                    )
                    schedulers = [
                        "DEISMultistepScheduler",
                        "HeunDiscreteScheduler",
                        "EulerDiscreteScheduler",
                        "DPMSolverMultistepScheduler",
                        "DPMSolverMultistepScheduler-Karras",
                        "DPMSolverMultistepScheduler-Karras-SDE",
                    ]
                    scheduler = gr.Dropdown(
                        label="Schedulers",
                        choices=schedulers,
                        value="EulerDiscreteScheduler",
                    )
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                    enhance_face_region = gr.Checkbox(label="Enhance non-face region", value=True)

            with gr.Column():
                gallery = gr.Gallery(label="Generated Images")
                usage_tips = gr.Markdown(
                    label="InstantID-Multi Usage Tips", value=tips, visible=False
                )
            face_files.upload(fn=swap_to_gallery, inputs=face_files, outputs=[face_uploaded_files, face_clear_button, face_files])
            face_remove_and_reupload.click(fn=remove_back_to_files, outputs=[face_uploaded_files, face_clear_button, face_files])

            pose_files.upload(fn=swap_to_gallery, inputs=pose_files, outputs=[pose_uploaded_files, pose_clear_button, pose_files])
            pose_remove_and_reupload.click(fn=remove_back_to_files, outputs=[pose_uploaded_files, pose_clear_button, pose_files])
            submit.click(
                fn=remove_tips,
                outputs=usage_tips,
            ).then(
                fn=randomize_seed_fn,
                inputs=[seed, randomize_seed],
                outputs=seed,
                queue=False,
                api_name=False,
            ).then(
                fn=generate_image,
                inputs=[
                    face_files,
                    pose_files,
                    prompt,
                    negative_prompt,
                    style,
                    num_steps,
                    identitynet_strength_ratio,
                    adapter_strength_ratio,
                    pose_strength,
                    canny_strength,
                    depth_strength,
                    controlnet_selection,
                    guidance_scale,
                    seed,
                    scheduler,
                    enable_LCM,
                    enhance_face_region,
                    mask_area,
                    denoise_strength,
                ],
                outputs=[gallery, usage_tips],
            )

            enable_LCM.input(
                fn=toggle_lcm_ui,
                inputs=[enable_LCM],
                outputs=[num_steps, guidance_scale, canny_strength, depth_strength,identitynet_strength_ratio,adapter_strength_ratio],
                queue=False,
            )



    demo.launch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path", type=str, default="wangqixun/YamerMIX_v8"
    )
    parser.add_argument(
        "--enable_LCM", type=bool, default=os.environ.get("ENABLE_LCM", False)
    )
    args = parser.parse_args()

    main(args.pretrained_model_name_or_path, args.enable_LCM)


