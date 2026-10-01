//! The OpenAI-compatible routes.
//!
//! - `POST /v1/audio/speech`: JSON in, audio out.
//! - `POST /v1/audio/transcriptions` and `/v1/audio/translations`:
//!   multipart upload in, JSON or text out.
//! - `GET /v1/models`, `GET /v1/models/{id}`.
//! - `GET /health`.
//!
//! There is deliberately nothing else: no web UI, no OpenAPI document, no
//! model management over HTTP (models are chosen when the server starts).

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use axum::extract::{DefaultBodyLimit, Path, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use tokio::sync::Semaphore;

use crate::error::ApiError;

/// Largest `/audio/speech` JSON body. 4096 characters of text fit easily.
pub const SPEECH_BODY_LIMIT: usize = 64 * 1024;
/// Largest upload, OpenAI's limit.
pub const UPLOAD_BODY_LIMIT: usize = 25 * 1024 * 1024;

/// One inference at a time per model (a GPU is serial anyway), with a
/// bounded line behind it. Past the line, callers get 503 and Retry-After
/// instead of piling up.
pub struct Gate {
    permits: Semaphore,
    waiting: AtomicUsize,
    max_waiting: usize,
}

impl Gate {
    pub fn new(max_waiting: usize) -> Self {
        Self { permits: Semaphore::new(1), waiting: AtomicUsize::new(0), max_waiting }
    }

    async fn enter(&self) -> Result<tokio::sync::SemaphorePermit<'_>, ApiError> {
        if self.waiting.fetch_add(1, Ordering::SeqCst) >= self.max_waiting {
            self.waiting.fetch_sub(1, Ordering::SeqCst);
            return Err(ApiError::busy(5));
        }
        let permit = self.permits.acquire().await;
        self.waiting.fetch_sub(1, Ordering::SeqCst);
        permit.map_err(|_| ApiError::internal("server shutting down"))
    }
}

pub struct AppState {
    pub engine: loqui::Engine,
    pub tts_gate: Gate,
    pub stt_gate: Gate,
    pub request_timeout: Duration,
}

pub fn router(state: Arc<AppState>) -> Router<()> {
    let router = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/models/{id}", get(get_model))
        .route("/v1/audio/speech", post(speech).layer(DefaultBodyLimit::max(SPEECH_BODY_LIMIT)));
    #[cfg(feature = "whisper")]
    let router = router
        .route("/v1/audio/transcriptions", post(stt::transcriptions).layer(DefaultBodyLimit::max(UPLOAD_BODY_LIMIT)))
        .route("/v1/audio/translations", post(stt::translations).layer(DefaultBodyLimit::max(UPLOAD_BODY_LIMIT)));
    router.with_state(state)
}

async fn health() -> Json<serde_json::Value> {
    // No versions, no model names: nothing to fingerprint.
    Json(serde_json::json!({ "status": "ok" }))
}

fn model_entries(engine: &loqui::Engine) -> Vec<serde_json::Value> {
    engine
        .models()
        .into_iter()
        .map(|m| {
            // `owned_by` ends in the task and `task` states it, which is how
            // OpenAI-style clients that route by model classify ids.
            let task = match m.kind {
                loqui::ModelKind::Tts => "tts",
                loqui::ModelKind::Stt => "stt",
            };
            serde_json::json!({ "id": m.id, "object": "model", "created": 0, "owned_by": format!("loqui/{task}"), "task": task })
        })
        .collect()
}

async fn list_models(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(serde_json::json!({ "object": "list", "data": model_entries(&state.engine) }))
}

async fn get_model(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Result<Json<serde_json::Value>, ApiError> {
    model_entries(&state.engine)
        .into_iter()
        .find(|m| m["id"] == id)
        .map(Json)
        .ok_or_else(|| ApiError::new(StatusCode::NOT_FOUND, "invalid_request_error", "model_not_found", "no such model"))
}

#[derive(Deserialize)]
struct SpeechBody {
    #[serde(default)]
    model: Option<String>,
    input: String,
    #[serde(default)]
    voice: Option<String>,
    #[serde(default)]
    response_format: Option<String>,
    #[serde(default)]
    speed: Option<f32>,
}

async fn speech(
    State(state): State<Arc<AppState>>,
    body: Result<Json<SpeechBody>, axum::extract::rejection::JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(body) = body.map_err(|e| {
        // Keep the rejection's own status: 413 for an oversized body, 415
        // for a missing JSON content type, 400/422 for malformed JSON.
        ApiError::new(e.status(), "invalid_request_error", "invalid_body", e.body_text())
    })?;
    if let Some(model) = &body.model
        && state.engine.stt_model_id() == Some(model.as_str())
    {
        return Err(ApiError::bad_request("wrong_model", "that is a transcription model"));
    }
    // OpenAI defaults to mp3; so does this server when built with it, and
    // falls back to WAV, the most widely playable, when not.
    let format = match body.response_format.as_deref() {
        Some(name) => loqui::Format::parse(name).map_err(|e| ApiError::bad_request("unsupported_format", e.to_string()))?,
        None => loqui::Format::DEFAULT,
    };
    let request = loqui::SpeakRequest {
        text: body.input,
        voice: body.voice.unwrap_or_else(|| loqui_kokoro::DEFAULT_VOICE.to_owned()),
        speed: body.speed.unwrap_or(1.0),
        format,
    };
    let chars = request.text.chars().count();
    let _permit = state.tts_gate.enter().await?;
    let started = Instant::now();
    let engine = state.engine.clone();
    let speech = run_blocking(state.request_timeout, move || engine.speak(&request)).await?;
    tracing::info!(chars, audio_secs = speech.duration_secs, elapsed_ms = started.elapsed().as_millis() as u64, "speech");
    Ok(([(header::CONTENT_TYPE, format.content_type())], speech.audio).into_response())
}

/// Runs blocking inference off the async runtime, with a deadline. The
/// work itself cannot be interrupted; past the deadline the caller gets 504
/// and the result is discarded.
async fn run_blocking<T: Send + 'static>(
    timeout: Duration,
    f: impl FnOnce() -> Result<T, loqui::Error> + Send + 'static,
) -> Result<T, ApiError> {
    match tokio::time::timeout(timeout, tokio::task::spawn_blocking(f)).await {
        Ok(Ok(result)) => result.map_err(ApiError::from),
        Ok(Err(join)) => Err(ApiError::internal(format!("inference task failed: {join}"))),
        Err(_) => Err(ApiError::new(StatusCode::GATEWAY_TIMEOUT, "server_error", "timeout", "the request took too long")),
    }
}

#[cfg(feature = "whisper")]
mod stt {
    use axum::extract::Multipart;

    use super::*;

    #[derive(Default)]
    struct Upload {
        file: Option<Vec<u8>>,
        model: Option<String>,
        language: Option<String>,
        prompt: Option<String>,
        response_format: Option<String>,
        temperature: Option<f32>,
    }

    async fn read_upload(mut form: Multipart) -> Result<Upload, ApiError> {
        let bad = |e: axum::extract::multipart::MultipartError| ApiError::bad_request("invalid_multipart", e.body_text());
        let mut upload = Upload::default();
        while let Some(field) = form.next_field().await.map_err(bad)? {
            let name = field.name().unwrap_or_default().to_owned();
            match name.as_str() {
                "file" => upload.file = Some(field.bytes().await.map_err(bad)?.to_vec()),
                "model" => upload.model = Some(field.text().await.map_err(bad)?),
                "language" => upload.language = Some(field.text().await.map_err(bad)?).filter(|l| !l.is_empty()),
                "prompt" => upload.prompt = Some(field.text().await.map_err(bad)?).filter(|p| !p.is_empty()),
                "response_format" => upload.response_format = Some(field.text().await.map_err(bad)?),
                "temperature" => {
                    let text = field.text().await.map_err(bad)?;
                    let t: f32 =
                        text.trim().parse().map_err(|_| ApiError::bad_request("invalid_temperature", "temperature must be a number"))?;
                    if !(0.0..=1.0).contains(&t) {
                        return Err(ApiError::bad_request("invalid_temperature", "temperature must be between 0 and 1"));
                    }
                    upload.temperature = Some(t);
                }
                // Unknown fields (timestamp_granularities, stream, ...) are
                // accepted and ignored, as OpenAI clients send them freely.
                _ => {}
            }
        }
        Ok(upload)
    }

    pub async fn transcriptions(State(state): State<Arc<AppState>>, form: Multipart) -> Result<Response, ApiError> {
        run(state, form, loqui::Task::Transcribe).await
    }

    pub async fn translations(State(state): State<Arc<AppState>>, form: Multipart) -> Result<Response, ApiError> {
        run(state, form, loqui::Task::Translate).await
    }

    async fn run(state: Arc<AppState>, form: Multipart, task: loqui::Task) -> Result<Response, ApiError> {
        let upload = read_upload(form).await?;
        if upload.model.as_deref() == Some(loqui::TTS_MODEL_ID) {
            return Err(ApiError::bad_request("wrong_model", "that is a speech synthesis model"));
        }
        let audio = upload.file.ok_or_else(|| ApiError::bad_request("missing_file", "the `file` field is required"))?;
        let format = upload.response_format.unwrap_or_else(|| "json".into());
        if !matches!(format.as_str(), "json" | "text" | "verbose_json" | "srt" | "vtt") {
            return Err(ApiError::bad_request("unsupported_format", "response_format must be json, text, verbose_json, srt or vtt"));
        }
        let request = loqui::TranscribeRequest {
            audio,
            language: if task == loqui::Task::Transcribe { upload.language } else { None },
            prompt: upload.prompt,
            task,
            temperature: upload.temperature.unwrap_or(0.0),
        };
        let bytes = request.audio.len();
        let _permit = state.stt_gate.enter().await?;
        let started = Instant::now();
        let engine = state.engine.clone();
        let result = run_blocking(state.request_timeout, move || engine.transcribe(&request)).await?;
        tracing::info!(
            upload_bytes = bytes,
            audio_secs = result.duration,
            elapsed_ms = started.elapsed().as_millis() as u64,
            "transcription"
        );
        Ok(render(&result, &format, task))
    }

    fn render(t: &loqui::Transcription, format: &str, task: loqui::Task) -> Response {
        let text_plain = [(header::CONTENT_TYPE, "text/plain; charset=utf-8")];
        match format {
            "text" => (text_plain, format!("{}\n", t.text)).into_response(),
            "srt" => (text_plain, subtitles(t, false)).into_response(),
            "vtt" => ([(header::CONTENT_TYPE, "text/vtt; charset=utf-8")], subtitles(t, true)).into_response(),
            "verbose_json" => Json(serde_json::json!({
                "task": if task == loqui::Task::Translate { "translate" } else { "transcribe" },
                "language": t.language,
                "duration": t.duration,
                "text": t.text,
                "segments": t.segments.iter().enumerate().map(|(i, s)| serde_json::json!({
                    "id": i, "start": s.start, "end": s.end, "text": s.text,
                })).collect::<Vec<_>>(),
            }))
            .into_response(),
            _ => Json(serde_json::json!({ "text": t.text })).into_response(),
        }
    }

    fn subtitles(t: &loqui::Transcription, vtt: bool) -> String {
        let stamp = |secs: f64| {
            let ms = (secs * 1000.0).round() as u64;
            let sep = if vtt { '.' } else { ',' };
            format!("{:02}:{:02}:{:02}{sep}{:03}", ms / 3_600_000, ms / 60_000 % 60, ms / 1000 % 60, ms % 1000)
        };
        let mut out = String::from(if vtt { "WEBVTT\n\n" } else { "" });
        for (i, s) in t.segments.iter().enumerate() {
            if !vtt {
                out.push_str(&format!("{}\n", i + 1));
            }
            out.push_str(&format!("{} --> {}\n{}\n\n", stamp(s.start), stamp(s.end), s.text.trim()));
        }
        out
    }
}
