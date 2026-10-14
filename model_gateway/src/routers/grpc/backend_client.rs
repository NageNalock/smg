//! Backend client: the polymorphism point over a worker's transport.
//!
//! A worker's backend is reached either via gRPC (the [`GrpcClient`] multiplexer
//! over SGLang/vLLM/TRT-LLM/MLX/TokenSpeed) or via a direct ZMQ connection to a
//! same-host engine — vLLM EngineCore or TokenSpeed ([`ZmqEngineClient`]).
//! `BackendClient` keeps those
//! first-class siblings — `GrpcClient` stays pure gRPC — while the execution
//! pipeline (which works against [`ProtoStream`]/[`ProtoGenerateRequest`]) is
//! shared unchanged.

use std::{collections::HashMap, ops::Deref, sync::Arc, time::Duration};

use openai_protocol::{
    chat::ChatCompletionRequest, completion::CompletionRequest, generate::GenerateRequest,
    messages::CreateMessageRequest, worker::WorkerLoadResponse,
};
use smg_grpc_client::{
    common_proto, sglang_proto, tokenizer_bundle::StreamBundle, tokenspeed_proto, vllm_proto,
    SglangGenerateRequestOptions, SglangSchedulerClient, TokenSpeedSchedulerClient,
    VllmEngineClient,
};
use tokio::sync::OnceCell;

use crate::{
    routers::grpc::{
        client::{
            GenerateRequestBuildOptions, GrpcClient, HealthCheckResponse, ModelInfo, ServerInfo,
        },
        common::stages::helpers,
        proto_wrapper::{
            finish_tokenspeed_request, finish_vllm_request, ProtoEmbedComplete, ProtoEmbedRequest,
            ProtoGenerateRequest, ProtoStream,
        },
        zmq_client::{self, fold_tokenizer_eos_backstop, ZmqDialect, ZmqEngineClient},
        MultimodalData,
    },
    worker::{ConnectionMode, RuntimeType},
};

/// A backend connection: gRPC (any engine) or direct ZMQ (vLLM EngineCore,
/// TokenSpeed or SGLang).
#[derive(Clone)]
pub enum BackendClient {
    Grpc(GrpcClient),
    Zmq(ZmqBackendClient),
}

/// Explicit control endpoints must refer to the same scheduler group as the
/// generation worker. Generation is always dispatched through `data`.
pub const ZMQ_CONTROL_URL_LABEL: &str = "zmq_control_url";
pub const ZMQ_ENCODE_URL_LABEL: &str = "zmq_encode_url";
pub const ZMQ_CONNECT_TIMEOUT_LABEL: &str = "zmq_connect_timeout_secs";

pub(crate) fn zmq_connect_timeout(labels: &HashMap<String, String>) -> Result<Duration, String> {
    let secs = labels
        .get(ZMQ_CONNECT_TIMEOUT_LABEL)
        .map(|value| {
            value
                .parse::<u64>()
                .map_err(|_| "zmq_connect_timeout_secs must be a positive integer".to_string())
        })
        .transpose()?
        .unwrap_or(600);
    if secs == 0 {
        return Err("zmq_connect_timeout_secs must be positive".to_string());
    }
    Ok(Duration::from_secs(secs))
}

/// The operations implemented by each direct transport. A control endpoint
/// does not imply the generation wire can carry KV-transfer rendezvous.
pub(crate) fn supports_pd(
    mode: ConnectionMode,
    runtime: RuntimeType,
    labels: &HashMap<String, String>,
) -> bool {
    mode == ConnectionMode::Grpc
        || (mode == ConnectionMode::Zmq
            && runtime == RuntimeType::TokenSpeed
            && labels.get("zmq_pd").is_some_and(|value| value == "true"))
}

fn explicit_grpc_endpoint(url: &str) -> Result<&str, String> {
    let normalized = url
        .strip_prefix("grpc://")
        .or_else(|| url.strip_prefix("grpcs://"))
        .ok_or_else(|| format!("control endpoint {url:?} must use grpc:// or grpcs://"))?;
    let parsed: http::Uri = format!("http://{normalized}")
        .parse()
        .map_err(|e| format!("invalid control endpoint {url:?}: {e}"))?;
    if parsed.host().is_none_or(str::is_empty)
        || parsed.port_u16().is_none()
        || parsed.path() != "/"
        || parsed.query().is_some()
    {
        return Err(format!(
            "control endpoint {url:?} must name a host and port without a path"
        ));
    }
    Ok(url)
}

pub(crate) fn control_endpoint(labels: &HashMap<String, String>) -> Result<Option<&str>, String> {
    labels
        .get(ZMQ_CONTROL_URL_LABEL)
        .map(|url| explicit_grpc_endpoint(url))
        .transpose()
}

pub(crate) fn encode_endpoint<'a>(
    mode: ConnectionMode,
    worker_url: &'a str,
    labels: &'a HashMap<String, String>,
) -> Result<&'a str, String> {
    match mode {
        ConnectionMode::Grpc => Ok(worker_url),
        ConnectionMode::Zmq => labels
            .get(ZMQ_ENCODE_URL_LABEL)
            .or_else(|| labels.get(ZMQ_CONTROL_URL_LABEL))
            .ok_or_else(|| {
                "ZMQ encode worker requires zmq_encode_url or zmq_control_url".to_string()
            })
            .and_then(|url| explicit_grpc_endpoint(url)),
        ConnectionMode::Http => {
            Err("typed encode dispatch requires a gRPC encoder endpoint".to_string())
        }
    }
}

#[derive(Clone)]
pub struct ZmqBackendClient {
    data: ZmqEngineClient,
    control_url: Option<String>,
    control: Arc<OnceCell<GrpcClient>>,
}

impl From<ZmqEngineClient> for ZmqBackendClient {
    fn from(data: ZmqEngineClient) -> Self {
        Self {
            data,
            control_url: None,
            control: Arc::new(OnceCell::new()),
        }
    }
}

impl Deref for ZmqBackendClient {
    type Target = ZmqEngineClient;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl ZmqBackendClient {
    pub(crate) fn with_control_endpoint(mut self, url: Option<String>) -> Self {
        if self.control_url != url {
            self.control = Arc::new(OnceCell::new());
            self.control_url = url;
        }
        self
    }

    async fn control(&self, operation: &str) -> Result<&GrpcClient, tonic::Status> {
        let endpoint = self.control_url.as_deref().ok_or_else(|| {
            tonic::Status::unimplemented(format!(
                "{operation} requires an explicit zmq_control_url for this ZMQ backend"
            ))
        })?;
        self.control
            .get_or_try_init(|| async {
                GrpcClient::connect(endpoint, self.runtime().as_str())
                    .await
                    .map_err(|e| {
                        tonic::Status::unavailable(format!(
                            "connect to control endpoint {endpoint}: {e}"
                        ))
                    })
            })
            .await
    }

    async fn flush_cache(
        &self,
        timeout_s: f32,
    ) -> Result<common_proto::FlushCacheResponse, tonic::Status> {
        // Use the native rank-broadcast utility where the engine implements it.
        let deadline = Duration::from_secs_f32(timeout_s.max(0.0))
            + smg_grpc_client::FLUSH_RPC_DEADLINE_MARGIN;
        let (success, message) = match self.runtime() {
            RuntimeType::Vllm => {
                let success = self.reset_prefix_cache(deadline).await?;
                (
                    success,
                    if success {
                        "Cache flushed successfully"
                    } else {
                        "Cache flush refused by an engine rank"
                    }
                    .to_string(),
                )
            }
            RuntimeType::Sglang => self.flush_cache_sglang(timeout_s, deadline).await?,
            _ => {
                return self
                    .control("FlushCache")
                    .await?
                    .flush_cache(timeout_s)
                    .await
            }
        };
        Ok(common_proto::FlushCacheResponse { success, message })
    }

    async fn profile(
        &self,
        req: Option<common_proto::StartProfileRequest>,
    ) -> Result<common_proto::ProfileResponse, tonic::Status> {
        if self.runtime() == RuntimeType::Sglang {
            let start = req.map(|req| engine_zmq_adapter::SglangProfileStart {
                output_dir: req.output_dir,
                start_step: req.start_step.map(i64::from),
                num_steps: req.num_steps.map(i64::from),
                activities: req.activities,
                with_stack: req.with_stack,
                record_shapes: req.record_shapes,
                profile_by_stage: req.profile_by_stage,
                profile_id: uuid::Uuid::now_v7().to_string(),
            });
            let (success, message) = self
                .profile_sglang(start, smg_grpc_client::PROFILE_RPC_DEADLINE)
                .await?;
            return Ok(common_proto::ProfileResponse { success, message });
        }
        let control = self.control("Profile").await?;
        match req {
            Some(req) => control.start_profile(req).await,
            None => control.stop_profile().await,
        }
    }
}

/// The native request builders of the ZMQ dialects for one request kind — the
/// only thing that differs between the ZMQ build surfaces, so the dialect
/// dispatch itself lives once in [`build_zmq_request`]/[`build_zmq_plain_request`].
struct ZmqBuilders<V, T, S> {
    vllm: V,
    tokenspeed: T,
    sglang: S,
}

/// A vLLM builder for a request kind carrying multimodal inputs and tool
/// constraints (chat, messages).
type VllmMmBuilder<B> = fn(
    String,
    &B,
    String,
    Vec<u32>,
    Option<vllm_proto::MultimodalInputs>,
    Option<(String, String)>,
) -> Result<vllm_proto::GenerateRequest, String>;

/// The TokenSpeed counterpart of [`VllmMmBuilder`].
type TokenSpeedMmBuilder<B> = fn(
    String,
    &B,
    String,
    Vec<u32>,
    Option<tokenspeed_proto::MultimodalInputs>,
    Option<(String, String)>,
) -> Result<tokenspeed_proto::GenerateRequest, String>;

/// The SGLang counterpart of [`VllmMmBuilder`]; its builders take the
/// multimodal inputs and constraints bundled as options.
type SglangMmBuilder<B> = fn(
    String,
    &B,
    String,
    Vec<u32>,
    SglangGenerateRequestOptions,
) -> Result<sglang_proto::GenerateRequest, String>;

/// A vLLM builder for a request kind with neither multimodal inputs nor tool
/// constraints (completion, plain generate); `T` is the builder's text
/// parameter.
type VllmPlainBuilder<B, T> =
    fn(String, &B, T, Vec<u32>) -> Result<vllm_proto::GenerateRequest, String>;

/// The TokenSpeed counterpart of [`VllmPlainBuilder`].
type TokenSpeedPlainBuilder<B, T> =
    fn(String, &B, T, Vec<u32>) -> Result<tokenspeed_proto::GenerateRequest, String>;

/// The SGLang counterpart of [`VllmPlainBuilder`].
type SglangPlainBuilder<B, T> =
    fn(String, &B, T, Vec<u32>) -> Result<sglang_proto::GenerateRequest, String>;

impl BackendClient {
    pub(crate) fn zmq_control_endpoint(&self) -> Option<&str> {
        match self {
            Self::Zmq(client) => client.control_url.as_deref(),
            Self::Grpc(_) => None,
        }
    }

    pub(crate) fn with_zmq_control_endpoint(mut self, url: Option<String>) -> Self {
        if let Self::Zmq(ref mut client) = self {
            *client = client.clone().with_control_endpoint(url);
        }
        self
    }

    /// Runtime type backing this client.
    pub fn runtime_type(&self) -> RuntimeType {
        match self {
            Self::Grpc(client) => client.runtime_type(),
            Self::Zmq(client) => client.runtime(),
        }
    }

    /// True if this is a direct-ZMQ backend (the engine receives token ids only
    /// and cannot match string stops itself).
    pub fn is_zmq(&self) -> bool {
        matches!(self, Self::Zmq(_))
    }

    /// Finalize a built generate request for this backend's wire: resolve
    /// string `stop`s the engine cannot match (token-only wires and SGLang's
    /// `skip_tokenizer_init` workers) into `stop_token_ids`, folding in EOS
    /// where the frontend owns stopping.
    ///
    /// Returns the router's residual obligation: the stop strings the engine
    /// will never see, which response processing must trim from output text.
    /// Empty when the engine matches stops server-side. This is the client's
    /// own policy — callers need no transport knowledge.
    pub fn finalize_generate_request(
        &self,
        request: &mut ProtoGenerateRequest,
        tokenizer: Option<&Arc<dyn llm_tokenizer::traits::Tokenizer>>,
    ) -> Vec<String> {
        let token_only_wire = self.is_zmq();
        let router_stops = helpers::resolve_string_stops(request, tokenizer, token_only_wire);
        if let Self::Zmq(client) = self {
            // EngineCore has no tokenizer, so stopping at EOS is this
            // frontend's job; TokenSpeed's scheduler stops at EOS itself, and
            // its requests are a different proto variant — the fold's own
            // variant match is the single dispatch point.
            //
            // The primary id must also ride the request's `_eos_token_id`, not
            // just the stop set, or an EOS finish surfaces as a matched_stop
            // token id. That field is filled from the client's own EOS set, so
            // hand it the tokenizer's ids for the deployments where the
            // connect-time model-dir lookup found none (the adopted set is
            // likewise only read by the vLLM translation).
            client.adopt_tokenizer_eos(tokenizer);
            fold_tokenizer_eos_backstop(request, tokenizer);
        }
        router_stops
    }

    /// Local liveness. gRPC has no cheap local flag (it uses a health RPC), so
    /// this reports `true` for gRPC; ZMQ reflects its connection liveness.
    pub fn is_alive(&self) -> bool {
        match self {
            Self::Grpc(_) => true,
            Self::Zmq(client) => client.is_alive(),
        }
    }

    /// Mutable SGLang client accessor. Only valid for a gRPC-SGLang backend;
    /// callers guard with a runtime/sglang check.
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees an SGLang gRPC backend"
    )]
    pub fn as_sglang_mut(&mut self) -> &mut SglangSchedulerClient {
        match self {
            Self::Grpc(client) => client.as_sglang_mut(),
            Self::Zmq(_) => panic!("Expected SGLang client, got ZMQ backend"),
        }
    }

    pub async fn health_check(&self) -> Result<HealthCheckResponse, tonic::Status> {
        match self {
            Self::Grpc(client) => client.health_check().await,
            Self::Zmq(client) => {
                let resp = client.health_check();
                Ok(HealthCheckResponse {
                    healthy: resp.healthy,
                    message: resp.message,
                })
            }
        }
    }

    pub async fn get_model_info(&self) -> Result<ModelInfo, tonic::Status> {
        match self {
            Self::Grpc(client) => client.get_model_info().await,
            Self::Zmq(client) => match client.control_url {
                Some(_) => client.control("GetModelInfo").await?.get_model_info().await,
                None => Ok(zmq_client::model_info(client)),
            },
        }
    }

    pub async fn get_server_info(&self) -> Result<ServerInfo, tonic::Status> {
        match self {
            Self::Grpc(client) => client.get_server_info().await,
            Self::Zmq(client) => match client.control_url {
                Some(_) => {
                    client
                        .control("GetServerInfo")
                        .await?
                        .get_server_info()
                        .await
                }
                None => Ok(zmq_client::server_info(client)),
            },
        }
    }

    pub async fn get_loads(&self) -> Result<WorkerLoadResponse, tonic::Status> {
        match self {
            Self::Grpc(client) => client.get_loads().await,
            Self::Zmq(client) => Ok(client.get_loads()),
        }
    }

    pub async fn flush_cache(
        &self,
        timeout_s: f32,
    ) -> Result<common_proto::FlushCacheResponse, tonic::Status> {
        match self {
            Self::Grpc(client) => client.flush_cache(timeout_s).await,
            Self::Zmq(client) => client.flush_cache(timeout_s).await,
        }
    }

    pub async fn start_profile(
        &self,
        req: common_proto::StartProfileRequest,
    ) -> Result<common_proto::ProfileResponse, tonic::Status> {
        match self {
            Self::Grpc(client) => client.start_profile(req).await,
            Self::Zmq(client) => client.profile(Some(req)).await,
        }
    }

    pub async fn stop_profile(&self) -> Result<common_proto::ProfileResponse, tonic::Status> {
        match self {
            Self::Grpc(client) => client.stop_profile().await,
            Self::Zmq(client) => client.profile(None).await,
        }
    }

    pub async fn subscribe_kv_events(
        &self,
        start_seq: u64,
    ) -> Result<tonic::Streaming<common_proto::KvEventBatch>, tonic::Status> {
        match self {
            Self::Grpc(client) => client.subscribe_kv_events(start_seq).await,
            Self::Zmq(_) => Err(tonic::Status::unimplemented(
                "SubscribeKvEvents not supported over ZMQ",
            )),
        }
    }

    pub async fn get_tokenizer(
        &self,
    ) -> Result<StreamBundle, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            Self::Grpc(client) => client.get_tokenizer().await,
            // EngineCore does not serve tokenizer artifacts over ZMQ; the
            // tokenizer is configured at worker registration instead.
            Self::Zmq(client) => client.control("GetTokenizer").await?.get_tokenizer().await,
        }
    }

    pub async fn generate(
        &mut self,
        req: ProtoGenerateRequest,
    ) -> Result<ProtoStream, tonic::Status> {
        match self {
            Self::Grpc(client) => client.generate(req).await,
            Self::Zmq(client) => Ok(ProtoStream::Zmq(zmq_client::generate(client, req).await?)),
        }
    }

    pub async fn embed(
        &mut self,
        req: ProtoEmbedRequest,
    ) -> Result<ProtoEmbedComplete, tonic::Status> {
        match self {
            Self::Grpc(client) => client.embed(req).await,
            Self::Zmq(_) => Err(tonic::Status::unimplemented(
                "ZMQ backend does not support embedding yet",
            )),
        }
    }

    pub fn build_chat_request(
        &self,
        request_id: String,
        body: &ChatCompletionRequest,
        processed_text: String,
        token_ids: Vec<u32>,
        options: GenerateRequestBuildOptions,
    ) -> Result<ProtoGenerateRequest, String> {
        match self {
            Self::Grpc(client) => {
                client.build_chat_request(request_id, body, processed_text, token_ids, options)
            }
            // A ZMQ backend speaks vLLM EngineCore or TokenSpeed directly; build
            // the native request for its dialect, mirroring the gRPC per-engine
            // dispatch in `GrpcClient::build_chat_request`.
            Self::Zmq(client) => build_zmq_request(
                client.dialect(),
                request_id,
                body,
                processed_text,
                token_ids,
                options,
                ZmqBuilders {
                    vllm: VllmEngineClient::build_generate_request_from_chat,
                    tokenspeed: TokenSpeedSchedulerClient::build_generate_request_from_chat,
                    sglang: SglangSchedulerClient::build_generate_request_from_chat_parts,
                },
            ),
        }
    }

    pub fn build_messages_request(
        &self,
        request_id: String,
        body: &CreateMessageRequest,
        processed_text: String,
        token_ids: Vec<u32>,
        options: GenerateRequestBuildOptions,
    ) -> Result<ProtoGenerateRequest, String> {
        match self {
            Self::Grpc(client) => {
                client.build_messages_request(request_id, body, processed_text, token_ids, options)
            }
            // Mirrors the gRPC per-engine dispatch: build the request natively for
            // the ZMQ backend's dialect (vLLM EngineCore or TokenSpeed).
            Self::Zmq(client) => build_zmq_request(
                client.dialect(),
                request_id,
                body,
                processed_text,
                token_ids,
                options,
                ZmqBuilders {
                    vllm: VllmEngineClient::build_generate_request_from_messages,
                    tokenspeed: TokenSpeedSchedulerClient::build_generate_request_from_messages,
                    sglang: SglangSchedulerClient::build_generate_request_from_messages_parts,
                },
            ),
        }
    }

    pub fn build_completion_request(
        &self,
        request_id: String,
        body: &CompletionRequest,
        original_text: String,
        token_ids: Vec<u32>,
    ) -> Result<ProtoGenerateRequest, String> {
        match self {
            Self::Grpc(client) => {
                client.build_completion_request(request_id, body, original_text, token_ids)
            }
            Self::Zmq(client) => build_zmq_plain_request(
                client.dialect(),
                request_id,
                body,
                original_text,
                token_ids,
                ZmqBuilders {
                    vllm: VllmEngineClient::build_generate_request_from_completion,
                    tokenspeed: TokenSpeedSchedulerClient::build_generate_request_from_completion,
                    sglang: SglangSchedulerClient::build_generate_request_from_completion_parts,
                },
            ),
        }
    }

    pub fn build_generate_request(
        &self,
        request_id: String,
        body: &GenerateRequest,
        original_text: Option<String>,
        token_ids: Vec<u32>,
    ) -> Result<ProtoGenerateRequest, String> {
        match self {
            Self::Grpc(client) => {
                client.build_generate_request(request_id, body, original_text, token_ids)
            }
            Self::Zmq(client) => build_zmq_plain_request(
                client.dialect(),
                request_id,
                body,
                original_text,
                token_ids,
                ZmqBuilders {
                    vllm: VllmEngineClient::build_plain_generate_request,
                    tokenspeed: TokenSpeedSchedulerClient::build_plain_generate_request,
                    sglang: SglangSchedulerClient::build_plain_generate_request_parts,
                },
            ),
        }
    }
}

/// Build a multimodal-carrying request (chat, messages) for a ZMQ backend: one
/// dispatch over the closed [`ZmqDialect`], converting the assembled multimodal
/// data to the dialect's proto and finishing through its SHM-cleanup wrapper.
fn build_zmq_request<B>(
    dialect: ZmqDialect,
    request_id: String,
    body: &B,
    processed_text: String,
    token_ids: Vec<u32>,
    options: GenerateRequestBuildOptions,
    builders: ZmqBuilders<VllmMmBuilder<B>, TokenSpeedMmBuilder<B>, SglangMmBuilder<B>>,
) -> Result<ProtoGenerateRequest, String> {
    let ZmqBuilders {
        vllm,
        tokenspeed,
        sglang,
    } = builders;
    match dialect {
        ZmqDialect::Vllm => {
            let vllm_mm = zmq_vllm_mm(options.multimodal_inputs)?;
            // The ZMQ wire carries one modality batch (see `zmq_vllm_mm`).
            finish_vllm_request(vllm_mm.map(|mm| (mm, Vec::new())), |mm| {
                vllm(
                    request_id,
                    body,
                    processed_text,
                    token_ids,
                    mm,
                    options.tool_constraints,
                )
            })
        }
        ZmqDialect::TokenSpeed => {
            let tokenspeed_mm = zmq_tokenspeed_mm(options.multimodal_inputs)?;
            finish_tokenspeed_request(tokenspeed_mm, |mm| {
                tokenspeed(
                    request_id,
                    body,
                    processed_text,
                    token_ids,
                    mm,
                    options.tool_constraints,
                )
            })
        }
        // SGLang tensors ride inline (no SHM handles to release on failure).
        ZmqDialect::Sglang => {
            let sglang_mm = zmq_sglang_mm(options.multimodal_inputs)?;
            let req = sglang(
                request_id,
                body,
                processed_text,
                token_ids,
                SglangGenerateRequestOptions {
                    multimodal_inputs: sglang_mm,
                    tool_call_constraint: options.tool_constraints,
                    require_reasoning: options.require_reasoning,
                },
            )?;
            Ok(ProtoGenerateRequest::Sglang(Box::new(req)))
        }
    }
}

/// Build a request kind with no multimodal or tool inputs (completion, plain
/// generate) for a ZMQ backend. `text` is the builders' shared text parameter
/// (`String` for completion, `Option<String>` for plain generate).
fn build_zmq_plain_request<B, T>(
    dialect: ZmqDialect,
    request_id: String,
    body: &B,
    text: T,
    token_ids: Vec<u32>,
    builders: ZmqBuilders<
        VllmPlainBuilder<B, T>,
        TokenSpeedPlainBuilder<B, T>,
        SglangPlainBuilder<B, T>,
    >,
) -> Result<ProtoGenerateRequest, String> {
    let ZmqBuilders {
        vllm,
        tokenspeed,
        sglang,
    } = builders;
    match dialect {
        ZmqDialect::Vllm => Ok(ProtoGenerateRequest::Vllm(Box::new(vllm(
            request_id, body, text, token_ids,
        )?))),
        ZmqDialect::TokenSpeed => Ok(ProtoGenerateRequest::TokenSpeed(Box::new(tokenspeed(
            request_id, body, text, token_ids,
        )?))),
        ZmqDialect::Sglang => Ok(ProtoGenerateRequest::Sglang(Box::new(sglang(
            request_id, body, text, token_ids,
        )?))),
    }
}

/// Convert assembled multimodal data for a vLLM ZMQ backend. A backend/variant
/// mismatch is a gateway bug (the assembly stage should produce the backend's
/// own variant), surfaced as a build error rather than a panic.
fn zmq_vllm_mm(
    inputs: Option<MultimodalData>,
) -> Result<Option<vllm_proto::MultimodalInputs>, String> {
    inputs
        .map(|mm| match mm {
            MultimodalData::Vllm(data) => {
                let (primary, extra) = data.into_protos();
                if !extra.is_empty() {
                    return Err(
                        "the vLLM ZMQ backend takes one modality per request; mixed image and \
                         video requests need the gRPC backend"
                            .to_string(),
                    );
                }
                Ok(primary)
            }
            other => Err(mm_variant_mismatch("vLLM", &other)),
        })
        .transpose()
}

/// Convert assembled multimodal data for a TokenSpeed ZMQ backend. See
/// [`zmq_vllm_mm`] for the mismatch semantics.
fn zmq_tokenspeed_mm(
    inputs: Option<MultimodalData>,
) -> Result<Option<tokenspeed_proto::MultimodalInputs>, String> {
    inputs
        .map(|mm| match mm {
            // No RDMA staging: the ZMQ engine reads tensors inline off the wire
            // and has no puller for `remote` payloads.
            MultimodalData::TokenSpeed(data) => Ok(data.into_proto(false)),
            other => Err(mm_variant_mismatch("TokenSpeed", &other)),
        })
        .transpose()
}

/// Convert assembled multimodal data for an SGLang ZMQ backend. See
/// [`zmq_vllm_mm`] for the mismatch semantics.
fn zmq_sglang_mm(
    inputs: Option<MultimodalData>,
) -> Result<Option<sglang_proto::MultimodalInputs>, String> {
    inputs
        .map(|mm| match mm {
            MultimodalData::Sglang(data) => Ok(data.into_proto()),
            other => Err(mm_variant_mismatch("SGLang", &other)),
        })
        .transpose()
}

/// Name the variant of a mismatched `MultimodalData` without dumping its tensor
/// payloads into the error string.
fn mm_variant_mismatch(expected: &str, got: &MultimodalData) -> String {
    let got = match got {
        MultimodalData::Sglang(_) => "SGLang",
        MultimodalData::Vllm(_) => "vLLM",
        MultimodalData::Trtllm(_) => "TRT-LLM",
        MultimodalData::TokenSpeed(_) => "TokenSpeed",
    };
    format!("multimodal data variant mismatch: {expected} ZMQ backend got {got} data")
}

#[cfg(test)]
mod capability_tests {
    use std::collections::HashMap;

    use super::*;
    use crate::worker::ConnectionMode;

    #[test]
    fn zmq_connect_timeout_is_validated_and_can_cover_model_loading() {
        assert_eq!(zmq_connect_timeout(&HashMap::new()).unwrap().as_secs(), 600);
        let labels = HashMap::from([(ZMQ_CONNECT_TIMEOUT_LABEL.to_string(), "7200".to_string())]);
        assert_eq!(zmq_connect_timeout(&labels).unwrap().as_secs(), 7200);
        for bad in ["0", "-1", "abc", "1.5"] {
            assert!(zmq_connect_timeout(&HashMap::from([(
                ZMQ_CONNECT_TIMEOUT_LABEL.to_string(),
                bad.to_string()
            )]))
            .is_err());
        }
    }

    #[test]
    fn zmq_pd_requires_a_runtime_with_transfer_support() {
        let labels = HashMap::from([("zmq_pd".to_string(), "true".to_string())]);
        assert!(supports_pd(
            ConnectionMode::Zmq,
            RuntimeType::TokenSpeed,
            &labels
        ));
        assert!(!supports_pd(
            ConnectionMode::Zmq,
            RuntimeType::TokenSpeed,
            &HashMap::new()
        ));
        assert!(!supports_pd(
            ConnectionMode::Zmq,
            RuntimeType::Vllm,
            &labels
        ));
        assert!(!supports_pd(
            ConnectionMode::Zmq,
            RuntimeType::Sglang,
            &labels
        ));
        assert!(!supports_pd(
            ConnectionMode::Zmq,
            RuntimeType::Unspecified,
            &labels
        ));
        assert!(supports_pd(
            ConnectionMode::Grpc,
            RuntimeType::Vllm,
            &HashMap::new()
        ));
    }

    #[test]
    fn encode_endpoint_uses_explicit_control_endpoint() {
        let mut labels = HashMap::from([(
            ZMQ_CONTROL_URL_LABEL.to_string(),
            "grpc://encoder:50051".to_string(),
        )]);
        assert_eq!(
            encode_endpoint(ConnectionMode::Zmq, "ipc:///tmp/encode", &labels).unwrap(),
            "grpc://encoder:50051"
        );
        labels.insert(
            ZMQ_ENCODE_URL_LABEL.to_string(),
            "grpcs://encoder:50052".to_string(),
        );
        assert_eq!(
            encode_endpoint(ConnectionMode::Zmq, "ipc:///tmp/encode", &labels).unwrap(),
            "grpcs://encoder:50052"
        );
        assert_eq!(
            encode_endpoint(ConnectionMode::Grpc, "grpc://encoder:50053", &labels).unwrap(),
            "grpc://encoder:50053"
        );
        assert!(
            encode_endpoint(ConnectionMode::Zmq, "ipc:///tmp/encode", &HashMap::new()).is_err()
        );
    }

    #[test]
    fn control_urls_are_explicit_and_validated() {
        for url in [
            "ipc:///tmp/engine",
            "http://worker:1",
            "worker:1",
            "grpc://",
        ] {
            let labels = HashMap::from([(ZMQ_CONTROL_URL_LABEL.to_string(), url.to_string())]);
            assert!(control_endpoint(&labels).is_err(), "{url}");
        }
        let labels = HashMap::from([(
            ZMQ_CONTROL_URL_LABEL.to_string(),
            "grpc://worker:1".to_string(),
        )]);
        assert_eq!(control_endpoint(&labels).unwrap(), Some("grpc://worker:1"));
    }
}

#[cfg(test)]
#[path = "backend_client_tests.rs"]
mod control_tests;
