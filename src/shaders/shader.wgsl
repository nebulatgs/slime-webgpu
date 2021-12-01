// Vertex shader
struct VertexInput {
    [[location(0)]] position: vec3<f32>;
};
struct VertexOutput {
    [[builtin(position)]] clip_position: vec4<f32>;
};

[[group(0), binding(0)]] var SourceTextureSampler : sampler;
[[group(0), binding(1)]] var SourceTexture : texture_storage_2d<rgba32float, read>;

[[stage(vertex)]]
fn vs_main(
    vertex: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(vertex.position, 1.0);
    return out;
}
 // Fragment shader

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    var thjing = textureLoad(SourceTexture, vec2<i32>(in.clip_position).xy);
    return thjing;
    // return in.clip_position / vec4<f32>(1000.0);
}
 