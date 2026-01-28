import sounddevice as sd
import numpy as np
import threading
import queue
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

# ===================== 核心配置（极简） =====================
SAMPLING_RATE = 16000  # Qwen3-ASR-Flash固定要求16000采样率
CHUNK_DURATION = 2     # 每2秒转写一次（可调，越小越实时）
MODEL_NAME = "Qwen/Qwen3-ASR-Flash"
LANGUAGE = "zh"        # 指定中文转写，提升准确率

# ===================== 初始化 =====================
# 加载Qwen3-ASR-Flash模型和处理器
print(f"正在加载{MODEL_NAME}模型...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_NAME)
model.eval()  # 设置为评估模式

# 音频队列：存储采集的音频数据
audio_queue = queue.Queue()

# ===================== 音频采集回调 =====================
def collect_audio(indata, frames, time, status):
    """麦克风采集回调，直接存原始音频数据"""
    if status:
        print(f"采集提示：{status}", flush=True)
    # 转换为Qwen3-ASR-Flash要求的格式（单声道、float32）
    audio_data = indata[:, 0].astype(np.float32)
    audio_queue.put(audio_data)

# ===================== 实时转写线程 =====================
def transcribe_real_time():
    """持续从队列取音频并转写，无ffmpeg依赖"""
    print(f"✅ 开始实时转写（{CHUNK_DURATION}秒/段），按Ctrl+C停止...")
    while True:
        # 收集指定时长的音频数据
        audio_chunks = []
        target_frames = int(SAMPLING_RATE * CHUNK_DURATION)  # 目标总帧数
        collected_frames = 0
        
        while collected_frames < target_frames:
            try:
                chunk = audio_queue.get(timeout=1)
                audio_chunks.append(chunk)
                collected_frames += len(chunk)
            except queue.Empty:
                break
        
        # 转写有效音频
        if audio_chunks:
            # 拼接并归一化（Qwen3-ASR-Flash必需步骤）
            audio = np.concatenate(audio_chunks)
            audio = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
            
            # 核心转写：使用Qwen3-ASR-Flash模型
            inputs = processor(audio, sampling_rate=SAMPLING_RATE, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(**inputs)
            result_text = processor.decode(outputs[0], skip_special_tokens=True)
            
            if result_text.strip():
                print(f"📝 转写结果：{result_text}")

# ===================== 启动程序 =====================
if __name__ == "__main__":
    # 启动转写线程
    transcribe_thread = threading.Thread(target=transcribe_real_time, daemon=True)
    transcribe_thread.start()
    
    # 启动麦克风采集（直接连接硬件驱动，无中间文件）
    with sd.InputStream(
        samplerate=SAMPLING_RATE,
        channels=1,  # 单声道
        callback=collect_audio,
        blocksize=1024  # 采集块大小，适配硬件
    ):
        try:
            input()  # 阻塞主线程，保持程序运行
        except KeyboardInterrupt:
            print("\n🛑 转写已停止")