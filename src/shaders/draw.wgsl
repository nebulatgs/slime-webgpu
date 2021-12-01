[[block]]struct ShaderParams {
    numAgents: f32;
    width: f32;
    height: f32;
    trailWeight: f32;
    deltaTime: f32;
    time: f32;
};
struct Agent {
	position: vec2<f32>;
	angle: f32;
};
[[block]]struct Agents {
    agents: array<Agent>;
};

[[group(0), binding(0)]] var<uniform> shaderParams : ShaderParams;
[[group(0), binding(1)]] var<storage, read> agents : Agents;
[[group(0), binding(2)]] var TargetTextureSampler : sampler;
[[group(0), binding(3)]] var TargetTexture : texture_storage_2d<rgba32float, write>;

[[stage(compute), workgroup_size(16,1,1)]]
fn main([[builtin(global_invocation_id)]] id: vec3<u32>) {
    if (id.x >= u32(shaderParams.numAgents)) {
		return;
	}

	let agent : Agent = agents.agents[id.x];
	// Draw to trail map
	let cellX : i32 = i32(agent.position.x);
	let cellY : i32 = i32(agent.position.y);

	textureStore(TargetTexture, vec2<i32>(cellX, cellY), vec4<f32>(0.1, 0.6, 0.5, 1.));
}