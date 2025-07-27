// Vertex shader
struct VertexInput {
    @location(0) position: vec3<f32>
};
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>
};

@group(0) @binding(0) var SourceTextureSampler : sampler;
@group(0) @binding(1) var SourceTexture : texture_storage_2d<rgba32float, read>;

@vertex
fn vs_main(
    vertex: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(vertex.position, 1.0);
    return out;
}

// Fragment shader

struct RenderParams {
    width: f32,
    height: f32,
    scaleDownFactor: f32,
};
@group(0) @binding(2) var<uniform> renderParams: RenderParams;

const GAMMA: f32 = 1.01;
const INV_GAMMA: f32 = 1.0 / GAMMA;

// Gamma correction
fn gamma_correct(color: f32) -> f32 {
    return pow(max(color, 0.0), INV_GAMMA);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var thing = textureLoad(SourceTexture, vec2<i32>((in.clip_position * renderParams.scaleDownFactor).xy));
    
    let corrected = gamma_correct(thing.r);
    
    return vec4<f32>(vec3<f32>(corrected * 10.0), 1.0);
    // return in.clip_position / vec4<f32>(1000.0);
}
 