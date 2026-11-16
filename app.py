import os, re, json, time, math, tempfile, subprocess, shutil, asyncio
from pathlib import Path

import streamlit as st
import requests
import edge_tts

APP_TITLE = "🎬 Myanmar Auto Video Recap"
MAX_MB = 500
GEMINI_MODEL = "gemini-3.8-flash"
GROQ_STT_MODEL = "whisper-large-v3-turbo"
OPENROUTER_MODEL = "openrouter/free"

st.set_page_config(page_title=APP_TITLE, page_icon="🎬", layout="centered")

def secret(name, default=""):
    try:
        return str(st.secrets.get(name, default) or default).strip()
    except Exception:
        return os.getenv(name, default).strip()

def run_cmd(args, timeout=900):
    p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError((p.stderr or p.stdout)[-2500:])
    return p.stdout.strip()

def ffmpeg_path():
    p = shutil.which("ffmpeg")
    if not p:
        raise RuntimeError("FFmpeg မတွေ့ပါ။ packages.txt ထဲတွင် ffmpeg ထည့်ထားကြောင်း စစ်ပါ။")
    return p

def extract_audio(video_path, audio_path):
    run_cmd([ffmpeg_path(), "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000",
             "-c:a", "mp3", "-b:a", "48k", str(audio_path)], timeout=1200)

def audio_chunks(audio_path, out_dir, chunk_seconds=480):
    out_dir.mkdir(parents=True, exist_ok=True)
    # 8-minute chunks stay comfortably below Groq's 25 MB free upload limit.
    run_cmd([ffmpeg_path(), "-y", "-i", str(audio_path), "-f", "segment",
             "-segment_time", str(chunk_seconds), "-reset_timestamps", "1",
             "-c:a", "libmp3lame", "-b:a", "48k", str(out_dir / "part_%03d.mp3")], timeout=1200)
    return sorted(out_dir.glob("part_*.mp3"))

def groq_transcribe(audio_path, api_key):
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}
    chunks_dir = audio_path.parent / "chunks"
    chunks = audio_chunks(audio_path, chunks_dir)
    transcripts = []
    for i, chunk in enumerate(chunks):
        with open(chunk, "rb") as f:
            r = requests.post(url, headers=headers,
                data={"model": GROQ_STT_MODEL, "response_format": "verbose_json"},
                files={"file": (chunk.name, f, "audio/mpeg")}, timeout=300)
        if r.status_code >= 400:
            raise RuntimeError(f"Groq STT error {r.status_code}: {r.text[:700]}")
        data = r.json()
        transcripts.append(data.get("text", ""))
    return "\n".join(t for t in transcripts if t.strip())

def local_whisper_transcribe(audio_path):
    # Fallback when Groq is unavailable. CPU processing can be slow on Streamlit Cloud.
    from faster_whisper import WhisperModel
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(audio_path), beam_size=3)
    return "\n".join(s.text.strip() for s in segments if s.text.strip())

def gemini_video_analysis(video_path, api_key, prompt):
    from google import genai
    client = genai.Client(api_key=api_key)
    uploaded = client.files.upload(file=str(video_path))
    deadline = time.time() + 900
    while time.time() < deadline:
        uploaded = client.files.get(name=uploaded.name)
        state = getattr(getattr(uploaded, "state", None), "name", "")
        if state == "ACTIVE":
            break
        if state == "FAILED":
            raise RuntimeError("Gemini Video File ပြင်ဆင်မှု မအောင်မြင်ပါ။")
        time.sleep(4)
    else:
        raise RuntimeError("Gemini Video ပြင်ဆင်ချိန် ကျော်လွန်သွားပါသည်။")
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[uploaded, prompt]
    )
    return response.text

def sample_frames(video_path, workdir, every_seconds=8, max_frames=36):
    frames_dir = workdir / "frames"
    frames_dir.mkdir(exist_ok=True)
    duration = float(run_cmd([ffmpeg_path(), "-i", str(video_path)], timeout=30) or 0) if False else None
    # ffprobe is included with FFmpeg on common deployments.
    probe = run_cmd(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)])
    duration = max(1, float(probe))
    step = max(every_seconds, math.ceil(duration / max_frames))
    paths = []
    for sec in range(0, int(duration), step):
        out = frames_dir / f"frame_{sec:06d}.jpg"
        run_cmd([ffmpeg_path(), "-y", "-ss", str(sec), "-i", str(video_path),
                 "-frames:v", "1", "-vf", "scale=768:-2", "-q:v", "5", str(out)], timeout=60)
        if out.exists():
            paths.append((sec, out))
    return paths, duration

def openrouter_vision_fallback(video_path, transcript, api_key, workdir):
    frames, duration = sample_frames(video_path, workdir)
    content = [{"type": "text", "text":
        "Analyze these sampled frames in chronological order. Return Burmese recap planning JSON. "
        "Include scene timestamps (seconds), visual descriptions, and a concise plot summary. "
        "Transcript (may be imperfect):\n" + transcript[:12000]}]
    for sec, path in frames:
        import base64
        encoded = base64.b64encode(path.read_bytes()).decode()
        content.append({"type": "text", "text": f"Timestamp: {sec} seconds"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}})
    payload = {"model": OPENROUTER_MODEL, "messages": [{"role": "user", "content": content}],
               "temperature": 0.3}
    r = requests.post("https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
                 "HTTP-Referer": "https://streamlit.io", "X-Title": "Myanmar Auto Video Recap"},
        json=payload, timeout=300)
    if r.status_code >= 400:
        raise RuntimeError(f"OpenRouter error {r.status_code}: {r.text[:900]}")
    return r.json()["choices"][0]["message"]["content"]

def make_burmese_script(transcript, analysis, target_seconds, api_keys):
    prompt = f"""
You are a Burmese movie-recap writer. Write a clear, engaging Burmese narration for a recap video.
Target duration: {target_seconds} seconds. Aim for natural spoken Burmese, not a literal translation.
Preserve the actual plot and chronological order. Do not invent events. Avoid filler and repeated lines.
Use the transcript and visual analysis below.
TRANSCRIPT:
{transcript[:16000]}
VIDEO ANALYSIS:
{analysis[:16000]}
Return ONLY the Burmese narration, with paragraphs separated by new lines.
"""
    gkey = api_keys.get("gemini")
    if gkey:
        try:
            from google import genai
            client = genai.Client(api_key=gkey)
            return client.models.generate_content(model=GEMINI_MODEL, contents=prompt).text.strip(), "Gemini"
        except Exception:
            pass
    okey = api_keys.get("openrouter")
    if okey:
        r = requests.post("https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {okey}", "Content-Type": "application/json"},
            json={"model": OPENROUTER_MODEL, "messages":[{"role":"user","content":prompt}],
                  "temperature":0.5}, timeout=180)
        if r.status_code < 400:
            return r.json()["choices"][0]["message"]["content"].strip(), "OpenRouter"
        raise RuntimeError(f"Recap text API error: {r.status_code} {r.text[:700]}")
    raise RuntimeError("Recap ရေးရန် GEMINI_API_KEY သို့မဟုတ် OPENROUTER_API_KEY လိုအပ်ပါသည်။")

async def tts_make(text, voice, output_path):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(output_path))

def make_srt(text, duration, srt_path):
    # Approximate sentence timing. It is a readable subtitle track, not forced word-level alignment.
    pieces = [p.strip() for p in re.split(r'(?<=[။!?])\s*|\n+', text) if p.strip()]
    if not pieces:
        pieces = [text.strip()]
    weights = [max(1, len(p)) for p in pieces]
    total = sum(weights)
    def stamp(sec):
        ms = int(sec * 1000)
        h, rem = divmod(ms, 3600000); m, rem = divmod(rem, 60000); s, ms = divmod(rem, 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"
    cursor = 0.0
    blocks = []
    for i, (p, w) in enumerate(zip(pieces, weights), 1):
        span = max(1.5, duration * w / total)
        end = min(duration, cursor + span)
        blocks.append(f"{i}\n{stamp(cursor)} --> {stamp(end)}\n{p}\n")
        cursor = end
    srt_path.write_text("\n".join(blocks), encoding="utf-8")

def render_video(source, voice_mp3, srt_path, out_path, vertical=True):
    duration_audio = float(run_cmd(["ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(voice_mp3)]))
    vf = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920" if vertical else "scale=1280:-2"
    # Use the original video as background; trim/loop only to match the narration length.
    # Subtitles are burned in with a portable ASS/SRT subtitle filter.
    srt_arg = str(srt_path).replace("\\", "/").replace(":", "\\:")
    run_cmd([ffmpeg_path(), "-y", "-stream_loop", "-1", "-i", str(source), "-i", str(voice_mp3),
        "-t", str(duration_audio), "-vf", f"{vf},subtitles='{srt_arg}'",
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "23", "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart",
        str(out_path)], timeout=3600)

def main():
    st.title(APP_TITLE)
    st.write("Video တစ်ခုတည်းတင်ပါ။ AI က ဇာတ်လမ်းကို နားလည်ပြီး မြန်မာလို Recap အသံဖန်တီးပေးပါမယ်။")
    with st.expander("API Keys / Provider ဆက်တင်", expanded=False):
        st.caption("Keys များကို Streamlit Secrets ထဲတွင် ထည့်ထားပါ။ ဒီနေရာတွင် မပြသပါ။")
        st.write("Gemini → Video analysis / Burmese script; Groq → speech-to-text; OpenRouter → fallback.")
    gemini_key = secret("GEMINI_API_KEY")
    groq_key = secret("GROQ_API_KEY")
    openrouter_key = secret("OPENROUTER_API_KEY")
    password = secret("APP_PASSWORD")
    if password:
        entered = st.text_input("ဝင်ရောက်ရန် Password", type="password")
        if entered != password:
            st.info("Password ထည့်ပြီးမှ ဆက်လုပ်နိုင်ပါမယ်။")
            st.stop()

    uploaded = st.file_uploader("Video တင်ပါ (MP4 / MOV / MKV)", type=["mp4","mov","mkv","webm"])
    target = st.select_slider("Recap အရှည်", options=["1 မိနစ်","2 မိနစ်","3 မိနစ်"], value="2 မိနစ်")
    voice_label = st.selectbox("မြန်မာအသံ", ["နီလာ (မိန်းကလေး)", "သီဟ (ယောက်ျားလေး)"])
    vertical = st.checkbox("9:16 Vertical Video (Shorts / TikTok)", value=True)
    if uploaded and uploaded.size > MAX_MB * 1024 * 1024:
        st.error(f"Video သည် {MAX_MB} MB ထက်ကျော်နေပါသည်။")
        st.stop()
    if st.button("🚀 မြန်မာ Recap ဖန်တီးမယ်", type="primary", disabled=uploaded is None):
        if not any([gemini_key, openrouter_key]):
            st.error("GEMINI_API_KEY သို့မဟုတ် OPENROUTER_API_KEY တစ်ခု အနည်းဆုံးလိုအပ်ပါတယ်။")
            st.stop()
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            video = work / uploaded.name
            video.write_bytes(uploaded.getbuffer())
            audio = work / "source.mp3"
            recap_mp3 = work / "myanmar_recap.mp3"
            srt = work / "myanmar_recap.srt"
            output = work / "myanmar_recap.mp4"
            bar = st.progress(0, text="Video ပြင်ဆင်နေသည်...")
            status = st.empty()
            try:
                extract_audio(video, audio)
                bar.progress(12, text="မူရင်းအသံ ထုတ်ယူပြီးပါပြီ")
                transcript = ""
                if groq_key:
                    try:
                        status.info("Groq API ဖြင့် Transcript ထုတ်ယူနေသည်...")
                        transcript = groq_transcribe(audio, groq_key)
                    except Exception as e:
                        status.warning(f"Groq မရပါ။ Local Whisper ကို စမ်းသုံးမည်။ ({str(e)[:180]})")
                if not transcript:
                    status.info("Local Whisper fallback ဖြင့် Transcript ထုတ်ယူနေသည်...")
                    transcript = local_whisper_transcribe(audio)
                if not transcript.strip():
                    raise RuntimeError("အသံ Transcript မရရှိပါ။")
                bar.progress(35, text="Transcript ပြီးပါပြီ")
                analysis = ""
                if gemini_key:
                    try:
                        status.info("Gemini ဖြင့် Video မြင်ကွင်းများကို စစ်ဆေးနေသည်...")
                        analysis = gemini_video_analysis(video, gemini_key,
                            "Analyze this video for a chronological movie recap. Identify the main plot, characters, "
                            "important turning points and scene timestamps. Do not invent events. Transcript:\n" + transcript[:10000])
                    except Exception as e:
                        status.warning(f"Gemini မရပါ။ OpenRouter fallback ကို စမ်းသုံးမည်။ ({str(e)[:180]})")
                if not analysis:
                    if not openrouter_key:
                        raise RuntimeError("Gemini မရပါ။ OpenRouter API Key မရှိသဖြင့် fallback မလုပ်နိုင်ပါ။")
                    analysis = openrouter_vision_fallback(video, transcript, openrouter_key, work)
                bar.progress(58, text="Video နားလည်မှု ပြီးပါပြီ")
                target_seconds = {"1 မိနစ်":60, "2 မိနစ်":120, "3 မိနစ်":180}[target]
                script, script_provider = make_burmese_script(transcript, analysis, target_seconds,
                    {"gemini": gemini_key, "openrouter": openrouter_key})
                bar.progress(68, text=f"မြန်မာ Script ရေးသားပြီးပါပြီ ({script_provider})")
                voice = "my-MM-NilarNeural" if voice_label.startswith("နီလာ") else "my-MM-ThihaNeural"
                asyncio.run(tts_make(script, voice, recap_mp3))
                audio_duration = float(run_cmd(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1", str(recap_mp3)]))
                make_srt(script, audio_duration, srt)
                bar.progress(82, text="မြန်မာအသံနှင့် စာတန်းထိုး ပြင်ဆင်ပြီးပါပြီ")
                render_video(video, recap_mp3, srt, output, vertical)
                bar.progress(100, text="ပြီးပါပြီ")
                status.success("မြန်မာ Recap ဖန်တီးပြီးပါပြီ။")
                st.subheader("မြန်မာ Recap Script")
                st.text_area("Script", script, height=220)
                st.video(str(output))
                st.download_button("⬇️ MP4 Download", output.read_bytes(),
                    file_name="myanmar_recap.mp4", mime="video/mp4")
                st.download_button("⬇️ MP3 Download", recap_mp3.read_bytes(),
                    file_name="myanmar_recap.mp3", mime="audio/mpeg")
                st.download_button("⬇️ SRT Download", srt.read_bytes(),
                    file_name="myanmar_recap.srt", mime="application/x-subrip")
                with st.expander("AI Analysis / Transcript"):
                    st.write(analysis)
                    st.text_area("Original Transcript", transcript, height=180)
            except Exception as e:
                bar.empty()
                st.error(f"လုပ်ဆောင်မှု မအောင်မြင်ပါ: {e}")
                st.info("API quota ပြည့်ခြင်း၊ Provider မရရှိခြင်း၊ သို့မဟုတ် Streamlit Cloud resource ကန့်သတ်ချက်ကြောင့် ဖြစ်နိုင်ပါတယ်။")

if __name__ == "__main__":
    main()
