import os,sys
now_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(now_dir)
import time
import folder_paths
import numpy as np
from PIL import Image
from skimage import img_as_ubyte
from huggingface_hub import snapshot_download
from fancyvideo.pipelines.fancyvideo_infer_pipeline import InferPipeline
pretrained_models = os.path.join(folder_paths.models_dir,"AIFSH","FancyVideo")
models_dir = os.path.join(pretrained_models,"resources","models")
out_dir = folder_paths.get_output_directory()

def load_models():
    if not os.path.exists(os.path.join(models_dir,"fancyvideo_ckpts/vae_3d_61_frames/mp_rank_00_model_states.pt")):
                snapshot_download(repo_id="qihoo360/FancyVideo",local_dir=pretrained_models,
                                ignore_patterns=["*md"])

                
    if not os.path.exists(os.path.join(models_dir,"stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin")):
        snapshot_download(repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
                        local_dir=os.path.join(models_dir,"stable-diffusion-v1-5"),
                        ignore_patterns=["v1-5*","*.safetensors","*fp16*","*non_ema*"])

class FancyVideoI2VNode:
    def __init__(self):
         load_models()
        
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "first_frame":("IMAGE",),
                "prompt":("TEXT",),
                "base_model":(["realisticVisionV60B1_v51VAE","toonyou_beta3","realcartoon3d_v15","pixarsRendermanInspo_mk1"],),
                "resolution":(["768*768","768*1024","1024*768"],),
                "video_length":([16,32],),
                "fps":("INT",{
                    "default":25
                }),
                "cond_motion_score":("FLOAT",{
                    "default":3.0
                }),
                "seed":("INT",{
                    "default":42
                }),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "gen_video"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_FancyVideo"

    def image2np(self,image,resolution):
       image_np = image.numpy()[0] * 255
       image_np = image_np.astype(np.uint8)
       image = Image.fromarray(image_np)
       image = image.resize((resolution[1],resolution[0]),)
       return np.array(image)

    def gen_video(self,first_frame,prompt,base_model,resolution,
                  video_length,fps,cond_motion_score,seed):
        text_to_video_mm_path = os.path.join(models_dir,f"fancyvideo_ckpts/vae_3d_{61 if video_length==16 else 125}_frames/mp_rank_00_model_states.pt")
        infer_pipeline = InferPipeline(
            text_to_video_mm_path=text_to_video_mm_path,
            base_model_path=os.path.join(models_dir,"sd_v1-5_base_models",f"{base_model}.safetensors"),
            res_adapter_type="res_adapter_v2",
            trained_keys=["motion_modules.", "conv_in.weight", "fps_embedding.", "motion_embedding."],
            model_path=models_dir,
            vae_type="vae_3d",
            use_fps_embedding=True,
            use_motion_embedding=True,
            common_positive_prompt="Best quality, masterpiece, ultra high res, photorealistic, Ultra realistic illustration, hyperrealistic, 8k",
            common_negative_prompt="(low quality:1.3), (worst quality:1.3),poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face,Facial blurring,a large crowd, many people,advertising, information, news, watermark, text, username, signature,out of frame, low res, error, cropped, worst quality, low quality, artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, nsfw, breast, naked, eroticism",
        )

        output_fps = fps # the fps to export video
        cond_fps = fps # condition fps
        cond_motion_score = cond_motion_score # condition motion score
        use_noise_scheduler_snr = True
        seed = seed
        
        dst_path = os.path.join(out_dir,f"fancyvideo_{time.time_ns()}.mp4")
        resolution_dict = {
            "768*768":(768,768),
            "768*1024":(1024,768),
            "1024*768":(768,1024)
        }
        resolution = resolution_dict[resolution]
        reference_image = self.image2np(first_frame,resolution)
        print(reference_image.shape)
        video = infer_pipeline.t2v_process_one_prompt(prompt=prompt, reference_image=reference_image, seed=seed, video_length=video_length, resolution=resolution, use_noise_scheduler_snr=use_noise_scheduler_snr, fps=cond_fps, motion_score=cond_motion_score,)
        frame_list = []
        for frame in video:
            frame = img_as_ubyte(frame.cpu().permute(1, 2, 0).float().detach().numpy())
            frame_list.append(frame)
        infer_pipeline.save_video(frame_list=frame_list, fps=output_fps, dst_path=dst_path)
        return (dst_path,)

class FancyVideoV2VNode:
    
    def __init__(self):
         load_models()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "video":("VIDEO",),
                "prompt":("TEXT",),
                "base_model":(["realisticVisionV60B1_v51VAE","toonyou_beta3","realcartoon3d_v15","pixarsRendermanInspo_mk1"],),
                "resolution":(["768*768","768*1024","1024*768"],),
                "infer_mode":(["extending","backtracking"],),
                "fps":("INT",{
                    "default":25
                }),
                "cond_motion_score":("FLOAT",{
                    "default":3.0
                }),
                "seed":("INT",{
                    "default":42
                }),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "gen_video"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_FancyVideo"


    def gen_video(self,video,prompt,base_model,resolution,
                  infer_mode,fps,cond_motion_score,seed):
        text_to_video_mm_path = os.path.join(models_dir,f"fancyvideo_ckpts/video_{infer_mode}/mp_rank_00_model_states.pt")
        infer_pipeline = InferPipeline(
            text_to_video_mm_path=text_to_video_mm_path,
            base_model_path=os.path.join(models_dir,"sd_v1-5_base_models",f"{base_model}.safetensors"),
            res_adapter_type="res_adapter_v2",
            trained_keys=["motion_modules.", "conv_in.weight", "fps_embedding.", "motion_embedding."],
            model_path=models_dir,
            vae_type="vae_3d",
            use_fps_embedding=True,
            use_motion_embedding=True,
            common_positive_prompt="Best quality, masterpiece, ultra high res, photorealistic, Ultra realistic illustration, hyperrealistic, 8k",
            common_negative_prompt="(low quality:1.3), (worst quality:1.3),poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face,Facial blurring,a large crowd, many people,advertising, information, news, watermark, text, username, signature,out of frame, low res, error, cropped, worst quality, low quality, artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, nsfw, breast, naked, eroticism"
        )

        output_fps = fps # the fps to export video
        cond_fps = fps # condition fps
        cond_motion_score = cond_motion_score # condition motion score
        use_noise_scheduler_snr = True
        seed = seed
        
        dst_path = os.path.join(out_dir,f"fancyvideo_{time.time_ns()}.mp4")
        resolution_dict = {
            "768*768":(768,768),
            "768*1024":(1024,768),
            "1024*768":(768,1024)
        }
        resolution = resolution_dict[resolution]
        
        video = infer_pipeline.video_expansion_process_one_prompt(infer_mode="video_"+infer_mode, prompt=prompt, reference_video_path=video, seed=seed, video_length=32, resolution=resolution, use_noise_scheduler_snr=use_noise_scheduler_snr, fps=cond_fps, motion_score=cond_motion_score,)
        frame_list = []
        for frame in video:
            frame = img_as_ubyte(frame.cpu().permute(1, 2, 0).float().detach().numpy())
            frame_list.append(frame)
        infer_pipeline.save_video(frame_list=frame_list, fps=output_fps, dst_path=dst_path)
        return (dst_path,)
    
NODE_CLASS_MAPPINGS = {
    "FancyVideoI2VNode": FancyVideoI2VNode,
    "FancyVideoV2VNode":FancyVideoV2VNode,
}




