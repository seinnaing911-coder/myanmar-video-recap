import os, re, json, time, tempfile, subprocess, shutil, asyncio
from pathlib import Path

import streamlit as st
import requests
import edge_tts

APP_TITLE = "🎬 Myanmar Auto Video Dubbing"
MAX_MB = 500
GROQ_STT_MODEL = "whisper-large-v3-turbo"
GEMINI_MODEL = "gemini-3.8-flash"
OPENROUTER_MODEL = "openrouter/free"
CHUNK_SECONDS = 480

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
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("FFmpeg မတွေ့ပါ။ packages.txt ထဲတွင် ffmpeg ထည့်ထားကြောင်း စစ်ပါ။")
    return path


def probe_duration(path):
    value = run_cmd(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", str(path)])
    return max(0.1, float(value))


def extract_audio(video_path, audio_path):
    # Original audio is extracted only for transcription. It is never copied to the final MP4.
    run_cmd([ffmpeg_path(), "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000",
             "-c:a", "libmp3lame", "-b:a", "48k", str(audio_path)], timeout=1800)


def split_audio(audio_path, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    run_cmd([ffmpeg_path(), "-y", "-i", str(audio_path), "-f", "segment",
             "-segment_time", str(CHUNK_SECONDS), "-reset_timestamps", "1",
             "-c:a", "libmp3lame", "-b:a", "48k", str(out_dir / "chunk_%03d.mp3")], timeout=1800)
    return sorted(out_dir.glob("chunk_*.mp3"))


def groq_transcribe(audio_path, api_key):
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}
    chunks = split_audio(audio_path, audio_path.parent / "audio_chunks")
    result = []
    for index, chunk in enumerate(chunks):
        with open(chunk, "rb") as file_obj:
            response = requests.post(
                url, headers=headers,
                data={"model": GROQ_STT_MODEL, "response_format": "verbose_json", "timestamp_granularities[]": "segment"},
                files={"file": (chunk.name, file_obj, "audio/mpeg")}, timeout=600)
        if response.status_code >= 400:
            raise RuntimeError(f"Groq STT error {response.status_code}: {response.text[:800]}")
        data = response.json()
        offset = index * CHUNK_SECONDS
        segments = data.get("segments") or []
        if segments:
            for item in segments:
                text = (item.get("text") or "").strip()
                if text:
                    result.append({"start": float(item.get("start", 0)) + offset,
                                   "end": float(item.get("end", item.get("start", 0) + 2)) + offset,
                                   "text": text})
        else:
            # Fallback for providers that return text but no segment timestamps.
            chunk_duration = probe_duration(chunk)
            text = (data.get("text") or "").strip()
            if text:
                result.extend(approximate_segments(text, offset, chunk_duration))
    return normalize_segments(result, probe_duration(audio_path))


def approximate_segments(text, offset, duration):
    parts = [p.strip() for p in re.split(r"(?<=[.!?。！？])\s+|\n+", text) if p.strip()]
    if not parts:
        parts = [text.strip()] if text.strip() else []
    if not parts:
        return []
    weights = [max(1, len(p)) for p in parts]
    total = sum(weights)
    cursor = 0.0
    result = []
    for part, weight in zip(parts, weights):
        span = max(1.0, duration * weight / total)
        result.append({"start": offset + cursor, "end": offset + min(duration, cursor + span), "text": part})
        cursor += span
    return result


def local_whisper_transcribe(audio_path):
    from faster_whisper import WhisperModel
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(audio_path), beam_size=3, vad_filter=True)
    return normalize_segments([{"start": s.start, "end": s.end, "text": s.text.strip()}
                               for s in segments if s.text.strip()], probe_duration(audio_path))


def normalize_segments(segments, duration):
    clean = []
    for item in sorted(segments, key=lambda x: float(x.get("start", 0))):
        text = re.sub(r"\s+", " ", str(item.get("text", ""))).strip()
        if not text:
            continue
        start = max(0.0, min(float(duration), float(item.get("start", 0))))
        end = max(start + 0.25, min(float(duration), float(item.get("end", start + 2))))
        if start >= duration:
            continue
        clean.append({"start": start, "end": end, "text": text})
    return clean


def parse_json_array(raw):
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I)
    try:
        data = json.loads(raw)
    except Exception:
        match = re.search(r"\[.*\]", raw, flags=re.S)
        if not match:
            raise RuntimeError("AI ဘာသာပြန်ရလဒ်ကို JSON အဖြစ် ဖတ်မရပါ။")
        data = json.loads(match.group(0))
    if isinstance(data, dict):
        data = data.get("translations") or data.get("items") or []
    if not isinstance(data, list):
        raise RuntimeError("AI ဘာသာပြန်ရလဒ်ပုံစံ မမှန်ပါ။")
    return data


def call_translation_api(prompt, provider, api_key):
    if provider == "Gemini":
        from google import genai
        client = genai.Client(api_key=api_key)
        return client.models.generate_content(model=GEMINI_MODEL, contents=prompt).text
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
                 "HTTP-Referer": "https://streamlit.io", "X-Title": "Myanmar Auto Video Dubbing"},
        json={"model": OPENROUTER_MODEL, "messages": [{"role": "user", "content": prompt}],
              "temperature": 0.2}, timeout=240)
    if response.status_code >= 400:
        raise RuntimeError(f"OpenRouter error {response.status_code}: {response.text[:700]}")
    return response.json()["choices"][0]["message"]["content"]


def translate_segments(segments, gemini_key, openrouter_key, progress_callback=None):
    translated = []
    batch_size = 18
    provider_order = []
    for start in range(0, len(segments), batch_size):
        batch = segments[start:start + batch_size]
        source_lines = [{"id": i, "text": seg["text"]} for i, seg in enumerate(batch)]
        prompt = (
            "Translate each line into natural, concise spoken Burmese for video dubbing. "
            "Preserve meaning, names, emotion and order. Do not add explanations or events. "
            "Keep each translation short enough to speak within its original time slot. "
            "Return ONLY a JSON array with objects containing id and myanmar. Keep every id.\n" +
            json.dumps(source_lines, ensure_ascii=False))
        raw = None
        errors = []
        if gemini_key:
            try:
                raw = call_translation_api(prompt, "Gemini", gemini_key)
                provider_order.append("Gemini")
            except Exception as exc:
                errors.append(f"Gemini: {str(exc)[:160]}")
        if raw is None and openrouter_key:
            try:
                raw = call_translation_api(prompt, "OpenRouter", openrouter_key)
                provider_order.append("OpenRouter")
            except Exception as exc:
                errors.append(f"OpenRouter: {str(exc)[:160]}")
        if raw is None:
            raise RuntimeError("မြန်မာဘာသာပြန် API မရပါ။ " + " | ".join(errors))
        rows = parse_json_array(raw)
        mapping = {}
        for row in rows:
            try:
                mapping[int(row.get("id"))] = str(row.get("myanmar", row.get("translation", ""))).strip()
            except Exception:
                continue
        for i, seg in enumerate(batch):
            burmese = mapping.get(i, "")
            if not burmese:
                # Preserve the timeline rather than silently dropping a segment.
                burmese = seg["text"]
            translated.append({**seg, "burmese": burmese})
        if progress_callback:
            progress_callback(min(1.0, (start + len(batch)) / len(segments)))
    provider = " / ".join(dict.fromkeys(provider_order)) or "AI"
    return translated, provider


async def tts_save(text, voice, path):
    await edge_tts.Communicate(text, voice).save(str(path))


def stamp(seconds):
    milliseconds = max(0, int(seconds * 1000))
    hours, rem = divmod(milliseconds, 3600000)
    minutes, rem = divmod(rem, 60000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{ms:03}"


def write_srt(segments, duration, path):
    blocks = []
    for index, seg in enumerate(segments, 1):
        start = min(duration, max(0, seg["start"]))
        end = min(duration, max(start + 0.25, seg["end"]))
        blocks.append(f"{index}\n{stamp(start)} --> {stamp(end)}\n{seg['burmese']}\n")
    path.write_text("\n".join(blocks), encoding="utf-8-sig")


def build_voice_track(segments, voice, workdir, duration, progress_callback=None):
    tts_dir = workdir / "tts_segments"
    tts_dir.mkdir(exist_ok=True)
    prepared = []
    for index, seg in enumerate(segments):
        out = tts_dir / f"voice_{index:04d}.mp3"
        asyncio.run(tts_save(seg["burmese"], voice, out))
        tts_duration = probe_duration(out)
        start = max(0.0, min(duration, seg["start"]))
        available = max(0.35, (segments[index + 1]["start"] - start) if index + 1 < len(segments) else duration - start)
        # Keep speech at a natural rate where possible. Speed up only when it exceeds the next speech slot.
        speed = min(1.5, max(1.0, tts_duration / available))
        adjusted = out
        if speed > 1.02:
            adjusted = tts_dir / f"voice_{index:04d}_fit.mp3"
            run_cmd([ffmpeg_path(), "-y", "-i", str(out), "-filter:a", f"atempo={speed:.3f}",
                     "-c:a", "libmp3lame", "-b:a", "96k", str(adjusted)], timeout=180)
        prepared.append((adjusted, int(start * 1000)))
        if progress_callback:
            progress_callback((index + 1) / len(segments))
    return prepared


def render_dubbed_video(source, prepared_audio, duration, srt_path, output_path, burn_subtitles):
    args = [ffmpeg_path(), "-y", "-i", str(source), "-f", "lavfi", "-t", str(duration),
            "-i", "anullsrc=r=24000:cl=mono"]
    for audio_path, _ in prepared_audio:
        args += ["-i", str(audio_path)]
    filters = [f"[1:a]atrim=duration={duration:.3f},asetpts=PTS-STARTPTS[base]"]
    mix_labels = ["[base]"]
    for i, (_, delay_ms) in enumerate(prepared_audio, start=2):
        label = f"v{i}"
        filters.append(f"[{i}:a]aresample=24000,adelay=delays={delay_ms}:all=1[{label}]")
        mix_labels.append(f"[{label}]")
    filters.append("".join(mix_labels) +
                   f"amix=inputs={len(mix_labels)}:duration=first:dropout_transition=0:normalize=0," +
                   f"atrim=duration={duration:.3f},asetpts=PTS-STARTPTS[aout]")
    video_filter = ""
    if burn_subtitles:
        srt_arg = str(srt_path).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
        video_filter = f"-vf subtitles='{srt_arg}'"
    args += ["-filter_complex", ";".join(filters), "-map", "0:v:0", "-map", "[aout]"]
    if burn_subtitles:
        args += ["-vf", f"subtitles='{srt_arg}'"]
    args += ["-t", f"{duration:.3f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
             "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", "-shortest", str(output_path)]
    run_cmd(args, timeout=7200)


def main():
    st.title(APP_TITLE)
    st.write("Video တစ်ခုတည်းတင်ပါ။ မူရင်းအသံအားလုံးကို ဖယ်ရှားပြီး မြန်မာအသံနဲ့ အစားထိုးပေးပါမယ်။")
    st.info("မူရင်းစကားပြောသံ၊ တေးဂီတ၊ ပတ်ဝန်းကျင်အသံအားလုံးကို Final MP4 ထဲ မထည့်ပါ။")

    gemini_key = secret("GEMINI_API_KEY")
    groq_key = secret("GROQ_API_KEY")
    openrouter_key = secret("OPENROUTER_API_KEY")
    password = secret("APP_PASSWORD")
    if password:
        entered = st.text_input("ဝင်ရောက်ရန် Password", type="password")
        if entered != password:
            st.info("Password ထည့်ပြီးမှ ဆက်လုပ်နိုင်ပါမယ်။")
            st.stop()

    uploaded = st.file_uploader("Video တင်ပါ (MP4 / MOV / MKV / WEBM)", type=["mp4", "mov", "mkv", "webm"])
    voice_label = st.selectbox("မြန်မာအသံရွေးချယ်ပါ", ["နီလာ (မိန်းကလေး)", "သီဟ (ယောက်ျားလေး)"])
    burn_subtitles = st.checkbox("မြန်မာစာတန်းထိုးကို Video ပေါ်တွင်ပါ ထည့်မယ်", value=True)
    st.caption("Video ကြာချိန်ကို မူရင်းအတိုင်း ထိန်းထားပါမယ်။ အသံအချိန်မကိုက်တဲ့ အပိုင်းတွေမှာ အသံအမြန်နှုန်းကို အနည်းငယ်ညှိနိုင်ပါတယ်။")

    if uploaded and uploaded.size > MAX_MB * 1024 * 1024:
        st.error(f"Video သည် {MAX_MB} MB ထက်ကျော်နေပါသည်။")
        st.stop()

    if st.button("🎙️ မြန်မာအသံနဲ့ Dubbing လုပ်မယ်", type="primary", disabled=uploaded is None):
        if not groq_key:
            st.error("GROQ_API_KEY လိုအပ်ပါတယ်။ Groq Whisper က မူရင်းအသံကို အချိန်မှတ်တမ်းနဲ့ စာသားဖော်ထုတ်ပေးပါမယ်။")
            st.stop()
        if not (gemini_key or openrouter_key):
            st.error("GEMINI_API_KEY သို့မဟုတ် OPENROUTER_API_KEY တစ်ခု လိုအပ်ပါတယ်။")
            st.stop()

        with tempfile.TemporaryDirectory() as temp_dir:
            work = Path(temp_dir)
            video = work / Path(uploaded.name).name
            video.write_bytes(uploaded.getbuffer())
            original_audio = work / "original_for_transcription.mp3"
            output_mp3 = work / "myanmar_voice.mp3"
            output_srt = work / "myanmar_subtitles.srt"
            output_mp4 = work / "myanmar_dubbed_video.mp4"
            duration = probe_duration(video)
            progress = st.progress(0, text="Video ပြင်ဆင်နေသည်... (0%)")
            status = st.empty()

            def set_progress(value, label):
                percent = int(max(0, min(100, value)))
                progress.progress(percent, text=f"{label} ({percent}%)")

            try:
                status.info("မူရင်းအသံကို စာသားဖော်ထုတ်ရန် ယာယီထုတ်ယူနေသည်...")
                extract_audio(video, original_audio)
                set_progress(8, "အသံဖိုင် ပြင်ဆင်ပြီးပါပြီ")

                status.info("Groq Whisper ဖြင့် မူရင်းစကားပြောသံနှင့် အချိန်မှတ်တမ်း ထုတ်ယူနေသည်...")
                try:
                    segments = groq_transcribe(original_audio, groq_key)
                except Exception as groq_error:
                    status.warning(f"Groq မအောင်မြင်ပါ။ Local Whisper ကို စမ်းသုံးနေသည်။ {str(groq_error)[:180]}")
                    segments = local_whisper_transcribe(original_audio)
                if not segments:
                    raise RuntimeError("မူရင်းစကားပြောသံမှ စာသားမရရှိပါ။ Video ထဲတွင် အသံရှိ/မရှိ စစ်ဆေးပါ။")
                set_progress(35, f"စကားပြောအပိုင်း {len(segments)} ခု ဖော်ထုတ်ပြီးပါပြီ")

                status.info("စာသားအပိုင်းတိုင်းကို မြန်မာလို ဘာသာပြန်နေသည်...")
                segments, translation_provider = translate_segments(
                    segments, gemini_key, openrouter_key,
                    lambda fraction: set_progress(35 + fraction * 25, "မြန်မာဘာသာပြန်နေသည်"))
                set_progress(62, f"မြန်မာဘာသာပြန်ပြီးပါပြီ ({translation_provider})")

                voice = "my-MM-NilarNeural" if voice_label.startswith("နီလာ") else "my-MM-ThihaNeural"
                status.info("မြန်မာအသံ အပိုင်းလိုက် ဖန်တီးပြီး မူရင်းအချိန်နဲ့ ညှိနေသည်...")
                prepared_audio = build_voice_track(
                    segments, voice, work, duration,
                    lambda fraction: set_progress(62 + fraction * 20, "မြန်မာအသံ ဖန်တီးနေသည်"))
                write_srt(segments, duration, output_srt)
                set_progress(84, "မြန်မာအသံနှင့် SRT ပြင်ဆင်ပြီးပါပြီ")

                status.info("မူရင်းအသံ Track ကို ဖယ်ရှားပြီး မြန်မာအသံတစ်ခုတည်း ထည့်နေသည်...")
                render_dubbed_video(video, prepared_audio, duration, output_srt, output_mp4, burn_subtitles)
                # Export a standalone MP3 from the final MP4 audio, ensuring it matches the final mix.
                run_cmd([ffmpeg_path(), "-y", "-i", str(output_mp4), "-vn", "-c:a", "libmp3lame",
                         "-b:a", "160k", str(output_mp3)], timeout=1200)
                set_progress(100, "ပြီးပါပြီ")
                status.success("Dubbing ပြီးပါပြီ။ Final MP4 တွင် မူရင်းအသံ Track မပါဝင်ပါ။")
                st.video(str(output_mp4))
                st.download_button("⬇️ မြန်မာအသံ Video (MP4)", output_mp4.read_bytes(),
                                   file_name="myanmar_dubbed_video.mp4", mime="video/mp4")
                st.download_button("⬇️ မြန်မာအသံ (MP3)", output_mp3.read_bytes(),
                                   file_name="myanmar_voice.mp3", mime="audio/mpeg")
                st.download_button("⬇️ မြန်မာစာတန်းထိုး (SRT)", output_srt.read_bytes(),
                                   file_name="myanmar_subtitles.srt", mime="application/x-subrip")
                with st.expander("Transcript / မြန်မာဘာသာပြန်စာသား"):
                    st.text_area("မူရင်း Transcript", "\n".join(s["text"] for s in segments), height=180)
                    st.text_area("မြန်မာ Script", "\n".join(s["burmese"] for s in segments), height=220)
            except Exception as exc:
                progress.empty()
                status.error(f"လုပ်ဆောင်မှု မအောင်မြင်ပါ: {exc}")
                st.caption("API quota၊ Network၊ FFmpeg သို့မဟုတ် Streamlit Cloud resource ကန့်သတ်ချက်ကြောင့် ဖြစ်နိုင်ပါတယ်။")


if __name__ == "__main__":
    main()
