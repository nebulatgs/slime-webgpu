struct Uniforms {
    projection: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var sim_texture: texture_2d<f32>;
@group(0) @binding(2) var sim_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = uniforms.projection * vec4<f32>(in.position.xy, 0.0, 1.0);
    out.tex_coords = vec2<f32>(
        in.position.x * 0.5 + 0.5,
        -(in.position.y * 0.5) + 0.5
    );
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(sim_texture, sim_sampler, in.tex_coords);
}
