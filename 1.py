import whisper
import sounddevice as sd
import numpy as np
import threading
import queue
import pyttsx3
import time

# ===================== 核心配置 =====================
SAMPLING_RATE = 16000  # Whisper固定要求16000采样率
CHUNK_DURATION = 5     # 每2秒转写一次（可调，越小越实时）
MODEL = "small"         # 模型大小：tiny(最快)/base(平衡)/small(更准)
LANGUAGE = "zh"        # 指定中文转写，提升准确率
KEYWORD = "導診助手"    # 触发关键词
RESPONSE_TEXT = "您好，请问需要导诊服务还是安全监护？"  # 回复内容

# ===================== 初始化 =====================
# 加载Whisper模型（首次运行自动下载到本地）
model = whisper.load_model(MODEL, device="cpu")  # 强制CPU，避免GPU依赖
# 音频队列：存储采集的音频数据
audio_queue = queue.Queue()
# 语音合成引擎初始化
engine = pyttsx3.init()
# 设置语音合成参数
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)  # 选择第一个语音（中文）
engine.setProperty('rate', 150)  # 设置语速
engine.setProperty('volume', 0.9)  # 设置音量

# 全局状态：是否已经回复过，避免重复触发
has_responded = False
response_lock = threading.Lock()

# ===================== 音频采集回调 =====================
def collect_audio(indata, frames, time, status):
    """麦克风采集回调，直接存原始音频数据"""
    if status:
        print(f"采集提示：{status}", flush=True)
    # 转换为Whisper要求的格式（单声道、float32）
    audio_data = indata[:, 0].astype(np.float32)
    audio_queue.put(audio_data)

# ===================== 语音回复函数 =====================
def speak_response():
    """语音回复函数，在独立线程中执行"""
    global has_responded
    with response_lock:
        if not has_responded:
            print(f"🎤 开始语音回复：{RESPONSE_TEXT}")
            engine.say(RESPONSE_TEXT)
            engine.runAndWait()
            has_responded = True
            # 30秒后重置回复状态，允许再次触发
            threading.Timer(30, reset_response_status).start()

def reset_response_status():
    """重置回复状态，允许再次触发"""
    global has_responded
    with response_lock:
        has_responded = False
    print("🔄 回复状态已重置，可以再次触发关键词")

# ===================== 实时转写与关键词检测线程 =====================
def transcribe_and_detect():
    """持续从队列取音频并转写，同时检测关键词"""
    print(f"✅ 开始实时转写与关键词检测（{CHUNK_DURATION}秒/段），按Ctrl+C停止...")
    print(f"🔍 等待关键词：'{KEYWORD}'")
    
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
            # 拼接并归一化（Whisper必需步骤）
            audio = np.concatenate(audio_chunks)
            audio = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
            
            # 核心转写：直接处理原始音频数组
            result = model.transcribe(
                audio,
                language=LANGUAGE,
                fp16=False,  # 关闭半精度，适配CPU
                verbose=False  # 关闭冗余日志
            )
            
            if result["text"].strip():
                print(f"📝 转写结果：{result['text']}")
                
                # 关键词检测
                with response_lock:
                    if KEYWORD in result["text"] and not has_responded:
                        print(f"🎉 检测到关键词：'{KEYWORD}'")
                        # 启动语音回复线程
                        threading.Thread(target=speak_response, daemon=True).start()

# ===================== 启动程序 =====================
if __name__ == "__main__":
    # 启动转写与检测线程
    transcribe_thread = threading.Thread(target=transcribe_and_detect, daemon=True)
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
            print("\n🛑 程序已停止")