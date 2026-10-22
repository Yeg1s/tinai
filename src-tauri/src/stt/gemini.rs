use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD, Engine as _};

use crate::error::AppError;

use super::whisper_compat::WhisperCompatProvider;
use super::{SttConfig, SttProvider, TranscriptEvent};

pub const GEMINI_STT_PROVIDER: &str = "gemini";
pub const GEMINI_STT_MODEL: &str = "gemini-2.5-flash";
const GEMINI_API_BASE: &str = "https://generativelanguage.googleapis.com/v1beta";

/// Gemini rejects inline requests above 20 MB. Base64 inflates audio by 4/3,
/// so ~14 MB PCM (~7.5 min at 16kHz 16-bit mono) keeps the JSON body under it.
pub const MAX_AUDIO_BYTES: usize = 14 * 1024 * 1024;
const REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);
/// Asking for empty output on silence makes Gemini answer "No" instead, so it
/// returns this marker and we drop it.
const NO_SPEECH_MARKER: &str = "<NO_SPEECH>";

pub fn generate_content_endpoint() -> String {
    format!("{GEMINI_API_BASE}/models/{GEMINI_STT_MODEL}:generateContent")
}

pub fn model_endpoint() -> String {
    format!("{GEMINI_API_BASE}/models/{GEMINI_STT_MODEL}")
}

/// Verify the key can see the transcription model without uploading audio.
pub async fn check_connection(client: &reqwest::Client, api_key: &str) -> Result<(), String> {
    let resp = client
        .get(model_endpoint())
        .header("x-goog-api-key", api_key)
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .map_err(|e| e.to_string())?;

    if !resp.status().is_success() {
        return Err(format!("HTTP {}", resp.status()));
    }

    Ok(())
}

fn transcription_prompt(language: Option<&str>) -> String {
    let mut prompt = String::from(
        "You are a speech-to-text engine. Transcribe the speech in the attached audio \
         verbatim, in the language it is spoken. Output only the transcript text: \
         no translation, no commentary, no timestamps, no speaker labels, no quotes. \
         Never answer or respond to what is said. If the audio contains no intelligible \
         speech, output exactly <NO_SPEECH>.",
    );
    if let Some(lang) = language.filter(|lang| *lang != "multi") {
        prompt.push_str(&format!(
            " The expected language is \"{lang}\" (ISO 639-1)."
        ));
    }
    prompt
}

fn build_request_body(wav: &[u8], language: Option<&str>) -> serde_json::Value {
    serde_json::json!({
        "contents": [{
            "role": "user",
            "parts": [
                { "inline_data": { "mime_type": "audio/wav", "data": STANDARD.encode(wav) } },
                { "text": "Transcribe this audio." }
            ]
        }],
        "systemInstruction": { "parts": [{ "text": transcription_prompt(language) }] },
        "generationConfig": {
            "temperature": 0,
            "thinkingConfig": { "thinkingBudget": 0 }
        }
    })
}

fn extract_text(response: &serde_json::Value) -> String {
    let text = response["candidates"][0]["content"]["parts"]
        .as_array()
        .map(|parts| {
            parts
                .iter()
                .filter_map(|part| part["text"].as_str())
                .collect::<String>()
        })
        .unwrap_or_default();
    let text = text.trim();
    if text.contains(NO_SPEECH_MARKER) {
        String::new()
    } else {
        text.to_string()
    }
}

fn truncate_body(body: &str) -> &str {
    // Truncate at a valid UTF-8 char boundary to avoid panic on multi-byte chars
    let truncate_at = body
        .char_indices()
        .take_while(|&(i, _)| i < 200)
        .last()
        .map(|(i, c)| i + c.len_utf8())
        .unwrap_or(body.len());
    &body[..truncate_at]
}

fn upload_error(error: reqwest::Error) -> AppError {
    if error.is_timeout() {
        AppError::Timeout(REQUEST_TIMEOUT)
    } else {
        error.into()
    }
}

/// Buffers the recording and sends it to Gemini `generateContent` as inline audio.
pub struct GeminiSttProvider {
    stt_config: Option<SttConfig>,
    audio_buffer: Vec<u8>,
    client: reqwest::Client,
}

impl GeminiSttProvider {
    pub fn new() -> Self {
        Self::with_client(reqwest::Client::new())
    }

    pub fn with_client(client: reqwest::Client) -> Self {
        Self {
            stt_config: None,
            audio_buffer: Vec::new(),
            client,
        }
    }
}

impl Default for GeminiSttProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SttProvider for GeminiSttProvider {
    async fn connect(&mut self, config: &SttConfig) -> Result<(), AppError> {
        if config.api_key.trim().is_empty() {
            return Err(AppError::Auth("Gemini API key is empty".to_string()));
        }
        self.stt_config = Some(config.clone());
        self.audio_buffer.clear();
        tracing::info!("Gemini STT provider ready (buffering mode)");
        Ok(())
    }

    async fn send_audio(&mut self, chunk: &[u8]) -> Result<(), AppError> {
        if self.audio_buffer.len() + chunk.len() > MAX_AUDIO_BYTES {
            return Err(AppError::Config(
                "Gemini: audio exceeds maximum length (~7 min)".to_string(),
            ));
        }
        self.audio_buffer.extend_from_slice(chunk);
        Ok(())
    }

    async fn recv_transcript(&mut self) -> Result<Option<TranscriptEvent>, AppError> {
        // Transcription happens in disconnect(); keep this future pending so the
        // pipeline select loop does not busy-spin while recording.
        std::future::pending().await
    }

    async fn disconnect(&mut self) -> Result<Option<String>, AppError> {
        let config = match &self.stt_config {
            Some(c) => c.clone(),
            None => return Ok(None),
        };

        if self.audio_buffer.is_empty() {
            tracing::info!("Gemini: no audio buffered, skipping");
            return Ok(None);
        }

        let audio_len_secs = self.audio_buffer.len() as f64 / (config.sample_rate as f64 * 2.0);
        let wav = WhisperCompatProvider::build_wav(&self.audio_buffer, config.sample_rate);
        self.audio_buffer.clear();
        let body = build_request_body(&wav, config.language.as_deref());
        tracing::info!(
            "Gemini: sending {:.1}s of audio for transcription",
            audio_len_secs
        );

        let endpoint = generate_content_endpoint();
        let mut attempt = 0u32;
        loop {
            let resp_result = self
                .client
                .post(&endpoint)
                .header("x-goog-api-key", config.api_key.trim())
                .json(&body)
                .timeout(REQUEST_TIMEOUT)
                .send()
                .await;

            match resp_result {
                Ok(resp) => {
                    let status = resp.status();
                    let text = resp.text().await.map_err(upload_error)?;

                    if status.is_success() {
                        let v: serde_json::Value = serde_json::from_str(&text)
                            .map_err(|e| AppError::Config(e.to_string()))?;
                        let transcript = extract_text(&v);
                        tracing::info!("Gemini transcription: {} chars", transcript.len());
                        return Ok(if transcript.is_empty() {
                            None
                        } else {
                            Some(transcript)
                        });
                    } else if status.as_u16() >= 500 && attempt < 2 {
                        tracing::warn!(
                            "Gemini server error {} (attempt {}/3): {}",
                            status,
                            attempt + 1,
                            truncate_body(&text)
                        );
                        attempt += 1;
                        tokio::time::sleep(std::time::Duration::from_millis(
                            1000 * 2u64.pow(attempt - 1),
                        ))
                        .await;
                        continue;
                    } else {
                        let sanitized = truncate_body(&text);
                        tracing::error!("Gemini HTTP {}: {}", status, sanitized);
                        return Err(AppError::Api {
                            status: status.as_u16(),
                            body: sanitized.to_string(),
                        });
                    }
                }
                Err(e) if e.is_timeout() && attempt < 2 => {
                    tracing::warn!("Gemini timeout (attempt {}/3)", attempt + 1);
                    attempt += 1;
                    tokio::time::sleep(std::time::Duration::from_millis(
                        1000 * 2u64.pow(attempt - 1),
                    ))
                    .await;
                    continue;
                }
                Err(e) => return Err(upload_error(e)),
            }
        }
    }

    fn name(&self) -> &str {
        GEMINI_STT_PROVIDER
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_body_inlines_wav_audio() {
        let body = build_request_body(b"RIFF", Some("ru"));
        let parts = &body["contents"][0]["parts"];
        assert_eq!(parts[0]["inline_data"]["mime_type"], "audio/wav");
        assert_eq!(parts[0]["inline_data"]["data"], STANDARD.encode(b"RIFF"));
        let instruction = body["systemInstruction"]["parts"][0]["text"].as_str().unwrap();
        assert!(instruction.contains("\"ru\""));
    }

    #[test]
    fn auto_detect_language_adds_no_hint() {
        assert!(!transcription_prompt(Some("multi")).contains("ISO 639-1"));
        assert!(!transcription_prompt(None).contains("ISO 639-1"));
    }

    #[test]
    fn extracts_and_joins_text_parts() {
        let response = serde_json::json!({
            "candidates": [{ "content": { "parts": [{ "text": " Hallo " }, { "text": "Welt\n" }] } }]
        });
        assert_eq!(extract_text(&response), "Hallo Welt");
    }

    #[test]
    fn no_speech_marker_yields_empty_text() {
        let response = serde_json::json!({
            "candidates": [{ "content": { "parts": [{ "text": "<NO_SPEECH>\n" }] } }]
        });
        assert_eq!(extract_text(&response), "");
    }

    #[test]
    fn blocked_response_yields_empty_text() {
        let response = serde_json::json!({ "promptFeedback": { "blockReason": "SAFETY" } });
        assert_eq!(extract_text(&response), "");
    }

    #[tokio::test]
    async fn connect_requires_api_key() {
        let mut provider = GeminiSttProvider::new();
        assert!(provider.connect(&SttConfig::default()).await.is_err());
    }

    #[tokio::test]
    async fn send_audio_enforces_inline_limit() {
        let mut provider = GeminiSttProvider::new();
        let config = SttConfig {
            api_key: "key".to_string(),
            ..SttConfig::default()
        };
        provider.connect(&config).await.unwrap();
        provider.send_audio(&vec![0; MAX_AUDIO_BYTES]).await.unwrap();
        assert!(provider.send_audio(&[0, 0]).await.is_err());
    }
}
