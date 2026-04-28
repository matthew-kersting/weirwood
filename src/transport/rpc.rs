//! Protocol Buffer message types used by the example TCP transport, defined
//! with `prost` attributes (no `protoc` dependency).
//!
//! These messages use the standard protobuf wire format and could be adapted
//! to gRPC in the future, but the bundled `examples/server.rs` and
//! `examples/client.rs` frame them directly over a raw TCP socket.

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct InitSessionRequest {
    #[prost(bytes, tag = "1")]
    pub server_key: ::prost::bytes::Bytes,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct InitSessionResponse {
    #[prost(string, tag = "1")]
    pub session_id: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PredictRequest {
    #[prost(string, tag = "1")]
    pub session_id: ::prost::alloc::string::String,

    #[prost(bytes, repeated, tag = "2")]
    pub features: ::prost::alloc::vec::Vec<::prost::alloc::vec::Vec<u8>>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PredictResponse {
    #[prost(bytes, tag = "1")]
    pub encrypted_score: ::prost::bytes::Bytes,
}
