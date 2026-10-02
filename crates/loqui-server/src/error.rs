//! Errors in OpenAI's shape: `{"error": {"message", "type", "code"}}`.
//! Messages describe the problem, never echo the caller's input, and never
//! include internal paths.

use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};

#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    kind: &'static str,
    code: &'static str,
    message: String,
    retry_after: Option<u64>,
}

impl ApiError {
    pub fn new(status: StatusCode, kind: &'static str, code: &'static str, message: impl Into<String>) -> Self {
        Self { status, kind, code, message: message.into(), retry_after: None }
    }

    pub fn bad_request(code: &'static str, message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "invalid_request_error", code, message)
    }

    /// The same response for a missing and a wrong token, so the two
    /// cannot be told apart.
    pub fn unauthorized() -> Self {
        Self::new(StatusCode::UNAUTHORIZED, "authentication_error", "invalid_api_key", "missing or invalid bearer token")
    }

    pub fn forbidden(code: &'static str, message: impl Into<String>) -> Self {
        Self::new(StatusCode::FORBIDDEN, "permission_error", code, message)
    }

    pub fn busy(retry_after: u64) -> Self {
        Self {
            retry_after: Some(retry_after),
            ..Self::new(StatusCode::SERVICE_UNAVAILABLE, "rate_limit_error", "server_busy", "the server is busy; retry shortly")
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, "server_error", "internal_error", message)
    }
}

impl From<loqui::Error> for ApiError {
    fn from(e: loqui::Error) -> Self {
        // Before the client-error check, which also counts a disabled
        // capability as the caller's doing: it is, but the answer is 404.
        if let loqui::Error::Disabled(what) = e {
            return Self::new(
                StatusCode::NOT_FOUND,
                "invalid_request_error",
                "not_enabled",
                format!("{what} is not enabled on this server"),
            );
        }
        if e.is_client_error() {
            return Self::bad_request("invalid_request", e.to_string());
        }
        // The detail can name cache paths; log it, return a summary.
        tracing::error!(error = %e, "request failed");
        Self::internal("the request could not be completed; see the server log")
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = serde_json::json!({
            "error": { "message": self.message, "type": self.kind, "code": self.code }
        });
        let mut response = (self.status, axum::Json(body)).into_response();
        if self.status == StatusCode::UNAUTHORIZED {
            response.headers_mut().insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
        }
        if let Some(secs) = self.retry_after {
            response.headers_mut().insert(header::RETRY_AFTER, HeaderValue::from(secs));
        }
        response
    }
}
