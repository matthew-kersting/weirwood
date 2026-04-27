//! Protocol Buffer message types for gRPC transport, defined using prost.
//!
//! This avoids the protoc dependency by defining types directly in Rust with prost attributes.

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
