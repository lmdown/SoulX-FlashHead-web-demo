import os
import gradio as gr
import torch
import random
import locale
import numpy as np
import librosa
import imageio
import subprocess
from collections import deque
from datetime import datetime
from loguru import logger
from PIL import Image
import tempfile
import time

from flash_head.inference import get_pipeline, get_base_data, get_infer_params, get_audio_embedding, run_pipeline

try:
    system_lang = locale.getlocale()[0] or locale.getdefaultlocale()[0] or ""
    system_lang_lower = system_lang.lower()
    if 'zh' in system_lang_lower or 'chinese' in system_lang_lower or 'cn' in system_lang_lower:
        current_lang = 'zh'
    else:
        current_lang = 'en'
except:
    current_lang = 'en'

I18N = {
    "title": {"zh": "# 🎬 FlashHead - 音频驱动视频生成", "en": "# 🎬 FlashHead - Audio Driven Video Generation"},
    "subtitle": {"zh": "上传图片和音频，生成音频驱动的视频", "en": "Upload image and audio to generate audio-driven video"},
    "cond_image": {"zh": "条件图片", "en": "Condition Image"},
    "audio_file": {"zh": "音频文件", "en": "Audio File"},
    "advanced_settings": {"zh": "高级设置", "en": "Advanced Settings"},
    "model_type": {"zh": "模型类型", "en": "Model Type"},
    "random_seed": {"zh": "随机种子", "en": "Random Seed"},
    "randomize": {"zh": "🎲 随机", "en": "🎲 Randomize"},
    "enable_face_crop": {"zh": "启用人脸裁剪", "en": "Enable Face Crop"},
    "model_dir": {"zh": "模型目录", "en": "Model Directory"},
    "wav2vec_dir": {"zh": "Wav2Vec目录", "en": "Wav2Vec Directory"},
    "generate_btn": {"zh": "🎥 生成视频", "en": "🎥 Generate Video"},
    "output_video": {"zh": "输出视频", "en": "Output Video"},
    "quick_examples": {"zh": "快速示例", "en": "Quick Examples"},
    "initializing": {"zh": "初始化模型...", "en": "Initializing model..."},
    "preparing": {"zh": "准备数据...", "en": "Preparing data..."},
    "loading_audio": {"zh": "加载音频...", "en": "Loading audio..."},
    "generating": {"zh": "开始生成视频...", "en": "Starting generation..."},
    "generating_progress": {"zh": "生成中 {}/{}，耗时: {:.2f}s", "en": "Generating {}/{}, elapsed: {:.2f}s"},
    "saving": {"zh": "保存视频...", "en": "Saving video..."},
    "finished": {"zh": "完成！", "en": "Finished!"},
    "upload_image": {"zh": "请上传条件图片", "en": "Please upload condition image"},
    "upload_audio": {"zh": "请上传音频文件", "en": "Please upload audio file"},
}

def _t(key):
    return I18N[key].get(current_lang, I18N[key].get('en', key))

def get_random_seed():
    return random.randint(0, 999999999)

pipeline = None
current_model_type = None

def initialize_pipeline(ckpt_dir, wav2vec_dir, model_type):
    global pipeline, current_model_type
    if pipeline is None or current_model_type != model_type:
        pipeline = get_pipeline(world_size=1, ckpt_dir=ckpt_dir, wav2vec_dir=wav2vec_dir, model_type=model_type)
        current_model_type = model_type
    return pipeline

def save_video_temp(frames_list, audio_path, fps):
    timestamp = datetime.now().strftime("%Y%m%d-%H-%M-%S-%f")[:-3]
    output_dir = 'sample_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    temp_video_path = os.path.join(output_dir, f"temp_{timestamp}.mp4")
    final_video_path = os.path.join(output_dir, f"res_{timestamp}.mp4")
    
    with imageio.get_writer(temp_video_path, format='mp4', mode='I',
                            fps=fps, codec='h264', ffmpeg_params=['-bf', '0']) as writer:
        for frames in frames_list:
            frames = frames.numpy().astype(np.uint8)
            for i in range(frames.shape[0]):
                frame = frames[i, :, :, :]
                writer.append_data(frame)
    
    cmd = ['ffmpeg', '-i', temp_video_path, '-i', audio_path, '-c:v', 'copy', '-c:a', 'aac', '-shortest', final_video_path, '-y']
    subprocess.run(cmd, capture_output=True)
    os.remove(temp_video_path)
    
    return final_video_path

def generate_video_gradio(cond_image, audio_file, model_type, base_seed, use_face_crop, ckpt_dir, wav2vec_dir, progress=gr.Progress()):
    if cond_image is None:
        raise gr.Error(_t("upload_image"))
    if audio_file is None:
        raise gr.Error(_t("upload_audio"))
    
    progress(0, desc=_t("initializing"))
    pipeline = initialize_pipeline(ckpt_dir, wav2vec_dir, model_type)
    
    progress(0.05, desc=_t("preparing"))
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        cond_image.save(tmp.name)
        cond_image_path = tmp.name
    
    get_base_data(pipeline, cond_image_path_or_dir=cond_image_path, base_seed=int(base_seed), use_face_crop=use_face_crop)
    os.unlink(cond_image_path)
    
    infer_params = get_infer_params()
    sample_rate = infer_params['sample_rate']
    tgt_fps = infer_params['tgt_fps']
    cached_audio_duration = infer_params['cached_audio_duration']
    frame_num = infer_params['frame_num']
    motion_frames_num = infer_params['motion_frames_num']
    slice_len = frame_num - motion_frames_num
    
    progress(0.1, desc=_t("loading_audio"))
    human_speech_array_all, _ = librosa.load(audio_file, sr=sample_rate, mono=True)
    
    cached_audio_length_sum = sample_rate * cached_audio_duration
    audio_end_idx = cached_audio_duration * tgt_fps
    audio_start_idx = audio_end_idx - frame_num
    
    audio_dq = deque([0.0] * cached_audio_length_sum, maxlen=cached_audio_length_sum)
    
    human_speech_array_slice_len = slice_len * sample_rate // tgt_fps
    total_slices = len(human_speech_array_all) // human_speech_array_slice_len
    human_speech_array_slices = human_speech_array_all[:total_slices * human_speech_array_slice_len].reshape(-1, human_speech_array_slice_len)
    
    generated_list = []
    progress(0.15, desc=_t("generating"))
    
    for chunk_idx, human_speech_array in enumerate(human_speech_array_slices):
        start_time = time.time()
        
        audio_dq.extend(human_speech_array.tolist())
        audio_array = np.array(audio_dq)
        audio_embedding = get_audio_embedding(pipeline, audio_array, audio_start_idx, audio_end_idx)
        
        video = run_pipeline(pipeline, audio_embedding)
        generated_list.append(video.cpu())
        
        elapsed = time.time() - start_time
        current_progress = 0.15 + (chunk_idx + 1) / total_slices * 0.75
        progress(current_progress, desc=_t("generating_progress").format(chunk_idx+1, total_slices, elapsed))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    
    progress(0.9, desc=_t("saving"))
    video_path = save_video_temp(generated_list, audio_file, fps=tgt_fps)
    
    progress(1.0, desc=_t("finished"))
    return video_path

custom_css = """
#audio_input .wrap {
    min-height: 70px !important;
}
"""

with gr.Blocks(title="FlashHead" if current_lang == 'en' else "FlashHead 视频生成") as demo:
    gr.Markdown(_t("title"))
    gr.Markdown(_t("subtitle"))
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                cond_image = gr.Image(type="pil", label=_t("cond_image"), height=350)
                audio_file = gr.Audio(type="filepath", label=_t("audio_file"), elem_id="audio_input")
            
            with gr.Accordion(_t("advanced_settings"), open=True):
                model_type = gr.Radio(["lite", "pro"], value="lite", label=_t("model_type"))
                
                with gr.Row():
                    base_seed = gr.Number(value=get_random_seed(), label=_t("random_seed"), precision=0)
                    randomize_btn = gr.Button(_t("randomize"), scale=0)
                
                use_face_crop = gr.Checkbox(value=True, label=_t("enable_face_crop"))
            
            ckpt_dir = gr.State(value="models/SoulX-FlashHead-1_3B")
            wav2vec_dir = gr.State(value="models/wav2vec2-base-960h")
        
        with gr.Column(scale=1):
            output_video = gr.Video(label=_t("output_video"), height=640, show_label=True)
            generate_btn = gr.Button(_t("generate_btn"), variant="primary")
    
    with gr.Row():
        with gr.Accordion(_t("quick_examples"), open=True):
            gr.Examples(
                examples=[
                    ["examples/girl.png", "examples/podcast_sichuan_16k.wav", "lite", 42, True],
                ],
                inputs=[cond_image, audio_file, model_type, base_seed, use_face_crop],
                label="",
            )
    
    randomize_btn.click(
        fn=get_random_seed,
        outputs=base_seed
    )
    
    generate_btn.click(
        fn=generate_video_gradio,
        inputs=[cond_image, audio_file, model_type, base_seed, use_face_crop, ckpt_dir, wav2vec_dir],
        outputs=output_video
    ).then(
        fn=get_random_seed,
        outputs=base_seed
    )

if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True, server_name="0.0.0.0", server_port=7861, css=custom_css)
