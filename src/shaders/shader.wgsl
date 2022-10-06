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

[[block]] struct RenderParams {
    width: f32;
    height: f32;
    scaleDownFactor: f32;
};
[[group(0), binding(2)]] var<uniform> renderParams: RenderParams;

// let col : vec4<f32> = vec4<f32>(0.00002436677, 0.98248443487, 0.31557875231, 1.);
let a : f32 = 2.;
[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    var thing = textureLoad(SourceTexture, vec2<i32>(in.clip_position * renderParams.scaleDownFactor).xy);
    
    return vec4<f32>(vec3<f32>((pow(a, thing.r) - 1.0) / (a - 1.0)), 1.0);
    // return in.clip_position / vec4<f32>(1000.0);
}
 