# Myanmar Auto Video Recap

## Included
- Streamlit single-video upload app
- Gemini video understanding as primary provider
- Groq Whisper transcription as primary speech-to-text provider
- OpenRouter free vision/chat model fallback
- Local faster-whisper fallback for transcription
- Edge TTS Burmese voices: Nilar / Thiha
- MP4, MP3 and SRT downloads

## Deploy
1. Upload `app.py`, `requirements.txt`, and `packages.txt` to a GitHub repository.
2. Deploy the repository on Streamlit Community Cloud.
3. In App Settings > Secrets, add values from `secrets.example.toml`.
4. Keep your API keys private. Do not paste them into chat or commit them to GitHub.

## Important limitations
- Free provider quotas and available models can change. A fallback only works if its key is configured and the provider has remaining quota.
- Gemini is the primary video-understanding route. OpenRouter fallback samples video frames and sends them with the transcript; it is not equivalent to full continuous video understanding.
- Groq's free speech-to-text upload limit is handled by splitting audio into chunks.
- Local faster-whisper fallback may be slow or exceed Streamlit Cloud memory/CPU limits.
- The current SRT timing is approximate sentence timing, not phoneme/word-level forced alignment.
- This first build keeps the original video as a looping background for the narration duration. It does not yet select and cut only the best story scenes; that requires a separate scene-selection/editing pass.
- Streamlit Community Cloud has resource/time limits; 500 MB upload support does not guarantee every 500 MB video can be processed successfully.
