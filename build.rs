// Compile the gRPC service definition under `proto/` only when the
// `transport` Cargo feature is enabled. Use the bundled `protoc` from
// `protoc-bin-vendored` so users do not need a system-installed protoc.

fn main() {
    if std::env::var_os("CARGO_FEATURE_TRANSPORT").is_none() {
        return;
    }

    if std::env::var_os("PROTOC").is_none() {
        let protoc = protoc_bin_vendored::protoc_bin_path()
            .expect("failed to locate vendored protoc binary");
        // Safe in single-threaded build.rs.
        unsafe {
            std::env::set_var("PROTOC", protoc);
        }
    }

    println!("cargo:rerun-if-changed=proto/inference.proto");
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(&["proto/inference.proto"], &["proto"])
        .expect("failed to compile proto/inference.proto");
}
