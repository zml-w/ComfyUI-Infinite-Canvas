import os
import folder_paths
from PIL import Image, ImageOps
import numpy as np
import torch
import torch.nn.functional as F
import cv2 # 必须确保安装 opencv-python
import imageio # 必须确保安装 imageio 和 imageio-ffmpeg

# 1. 文本输入节点
class CanvasTextInput:
    @classmethod
    def INPUT_TYPES(s): return {"required": {"text": ("STRING", {"multiline": True, "default": ""})}}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = "无限画布"
    def process(self, text): return (text,)

# 2. 图像输入节点
class CanvasImageInput:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required": {"image": (sorted(files), {"image_upload": True})}}
    CATEGORY = "无限画布"
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load"
    def load(self, image):
        path = folder_paths.get_annotated_filepath(image)
        i = Image.open(path).convert("RGB")
        img = np.array(i).astype(np.float32) / 255.0
        img = torch.from_numpy(img)[None,]
        return (img, torch.zeros((64,64), dtype=torch.float32, device="cpu").unsqueeze(0))

# 3. 视频输入节点
class CanvasVideoInput:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(('.mp4', '.mov', '.avi', '.webm', '.mkv', '.gif'))]
        return {
            "required": {
                "video": (sorted(files), {"image_upload": True}),
                "frame_limit": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1, "display": "number"}), 
            }
        }
    CATEGORY = "无限画布"
    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("images", "frame_count", "fps")
    FUNCTION = "load_video"

    def load_video(self, video, frame_limit):
        path = folder_paths.get_annotated_filepath(video)
        cap = cv2.VideoCapture(path)
        frames = []
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0: fps = 24
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            # cv2 reads BGR, convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frames.append(torch.from_numpy(frame))
            count += 1
            if frame_limit > 0 and count >= frame_limit: break
        cap.release()
        if len(frames) == 0: return (torch.zeros((1, 512, 512, 3)), 0, fps)
        return (torch.stack(frames), len(frames), fps)

# 4. 图像输出节点
class CanvasImageOutput:
    def __init__(self): self.output_dir = folder_paths.get_output_directory()
    @classmethod
    def INPUT_TYPES(s): return {"required": {"images": ("IMAGE", )}}
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "无限画布"
    def save(self, images):
        results = []
        for img in images:
            i = 255.0 * img.cpu().numpy()
            im = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            file = f"Canvas_Img_{len(os.listdir(self.output_dir))}.png"
            im.save(os.path.join(self.output_dir, file))
            results.append({"filename": file, "subfolder": "", "type": "output"})
        return { "ui": { "images": results } }

# 5. 视频输出节点 (使用 OpenCV 强制合成 MP4)
class CanvasVideoOutput:
    def __init__(self): self.output_dir = folder_paths.get_output_directory()
    @classmethod
    def INPUT_TYPES(s): 
        return {
            "required": {
                "images": ("IMAGE", ),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
            }
        }
    RETURN_TYPES = ()
    FUNCTION = "save_video"
    OUTPUT_NODE = True
    CATEGORY = "无限画布"

    def save_video(self, images, fps):
        # images: [Batch, Height, Width, Channels]
        # 转换为 numpy uint8 [0-255]
        video_np = (255.0 * images.cpu().numpy()).astype(np.uint8)
        
        filename = f"Canvas_Vid_{len(os.listdir(self.output_dir))}.mp4"
        filepath = os.path.join(self.output_dir, filename)

        try:
            # 使用 imageio 库进行视频编码，利用 imageio-ffmpeg 插件
            # 编码器设置为 'libx264' 以强制使用 H.264
            imageio.mimwrite(filepath, video_np, fps=fps, codec='libx264', quality=8) # quality 0-10, 10 is best
                
        except Exception as e:
            print(f"CanvasVideoOutput Error with imageio: {e}")
            return { "ui": { "images": [] } }

        # 返回 UI 信息，让前端知道这是一个 output 文件
        return { "ui": { "images": [{"filename": filename, "subfolder": "", "type": "output"}] } }

# 6. 画布媒体处理
class CanvasResolutionSelector:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "max_resolution": (["Disabled", "512", "768", "1024", "1280", "1536", "2048"], {"default": "Disabled"}),
                "aspect_ratio": (["Disabled", "1:1", "4:3", "3:4", "16:9", "9:16", "21:9", "9:21"], {"default": "Disabled"}),
                "method": (["Center Crop", "Stretch", "Fill White"], {"default": "Center Crop"}),
            },
            "optional": { "image": ("IMAGE",) }
        }
    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "process_image"
    CATEGORY = "无限画布"

    def process_image(self, max_resolution, aspect_ratio, method, image=None):
        if image is None:
            w, h = 1024, 1024
            if max_resolution != "Disabled": w = h = int(max_resolution)
            if aspect_ratio != "Disabled":
                wr, hr = map(int, aspect_ratio.split(":"))
                if wr > hr: h = int(w * hr / wr)
                else: w = int(h * wr / hr)
            empty = torch.zeros((1, h, w, 3), dtype=torch.float32)
            return (empty, w, h)

        batch_size, oh, ow, _ = image.shape
        target_w, target_h = ow, oh

        if max_resolution != "Disabled":
            max_res = int(max_resolution)
            scale = max_res / max(ow, oh)
            target_w = int(ow * scale)
            target_h = int(oh * scale)

        if aspect_ratio != "Disabled":
            wr, hr = map(int, aspect_ratio.split(":"))
            target_ratio = wr / hr
            current_ratio = target_w / target_h
            if abs(current_ratio - target_ratio) > 0.01:
                if method == "Stretch":
                    if target_ratio > 1: target_h = int(target_w / target_ratio)
                    else: target_w = int(target_h * target_ratio)
                elif method == "Center Crop":
                    if current_ratio > target_ratio: target_w = int(target_h * target_ratio)
                    else: target_h = int(target_w / target_ratio)
                elif method == "Fill White":
                    if current_ratio > target_ratio: target_h = int(target_w / target_ratio)
                    else: target_w = int(target_h * target_ratio)

        target_w = (target_w // 8) * 8
        target_h = (target_h // 8) * 8
        ret_image = image.permute(0, 3, 1, 2)

        if method == "Stretch" and aspect_ratio != "Disabled":
             ret_image = F.interpolate(ret_image, size=(target_h, target_w), mode="bilinear", align_corners=False)
        else:
            if method == "Center Crop": scale = max(target_w / ow, target_h / oh)
            elif method == "Fill White": scale = min(target_w / ow, target_h / oh)
            else: scale = target_w / ow
            content_w, content_h = int(ow * scale), int(oh * scale)
            ret_image = F.interpolate(ret_image, size=(content_h, content_w), mode="bilinear", align_corners=False)

            if method == "Center Crop":
                start_x, start_y = max(0, (content_w - target_w) // 2), max(0, (content_h - target_h) // 2)
                ret_image = ret_image[:, :, start_y:start_y+target_h, start_x:start_x+target_w]
            elif method == "Fill White":
                bg = torch.ones((batch_size, 3, target_h, target_w), dtype=torch.float32, device=ret_image.device)
                paste_x, paste_y = max(0, (target_w - content_w) // 2), max(0, (target_h - content_h) // 2)
                h_slice, w_slice = min(content_h, target_h - paste_y), min(content_w, target_w - paste_x)
                bg[:, :, paste_y:paste_y+h_slice, paste_x:paste_x+w_slice] = ret_image[:, :, :h_slice, :w_slice]
                ret_image = bg

        ret_image = ret_image.permute(0, 2, 3, 1)
        return (ret_image, target_w, target_h)

NODE_CLASS_MAPPINGS = {
    "CanvasTextInput": CanvasTextInput,
    "CanvasImageInput": CanvasImageInput,
    "CanvasImageOutput": CanvasImageOutput,
    "CanvasResolutionSelector": CanvasResolutionSelector,
    "CanvasVideoInput": CanvasVideoInput,
    "CanvasVideoOutput": CanvasVideoOutput
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CanvasTextInput": "画布文本输入 📝",
    "CanvasImageInput": "画布图像输入 🖼️",
    "CanvasImageOutput": "画布图像输出 📤",
    "CanvasResolutionSelector": "画布媒体处理 & 分辨率 📐",
    "CanvasVideoInput": "画布视频输入 🎬",
    "CanvasVideoOutput": "画布视频输出 🎬"
}
