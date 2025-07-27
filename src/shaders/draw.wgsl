[[block]]struct ShaderParams {
    numAgents: f32;
    width: f32;
    height: f32;
    trailWeight: f32;
    deltaTime: f32;
    time: f32;
};
struct Agent {
	posX: f32;
	posY: f32;
	angle: f32;
	//intensity: f32;
};
[[block]]struct Agents {
    agents: array<Agent>;
};

//let col : vec4<f32> = vec4<f32>(pow(0.008, 2.2), pow(0.992, 2.2), pow(0.592, 2.2), 1.);
//let col : vec4<f32> = vec4<f32>(0.00002436677, 0.98248443487, 0.31557875231, 1.);
//let col : vec4<f32> = vec4<f32>(0.01, 0.01, 0.01, 1.0);

[[group(0), binding(0)]] var<uniform> shaderParams : ShaderParams;
[[group(0), binding(1)]] var<storage, read> agents : Agents;
[[group(0), binding(2)]] var TargetTextureSampler : sampler;
[[group(0), binding(3)]] var TargetTexture : texture_storage_2d<rgba32float, write>;

[[stage(compute), workgroup_size(2,1,1)]]
fn main([[builtin(global_invocation_id)]] id: vec3<u32>) {
    if (id.x >= u32(shaderParams.numAgents)) {
		return;
	}

	let agent : Agent = agents.agents[id.x];
	// Draw to trail map
	let cellX : i32 = i32(agent.posX);
	let cellY : i32 = i32(agent.posY);

//	textureStore(TargetTexture, vec2<i32>(cellX, cellY), col);
//	textureStore(TargetTexture, vec2<i32>(cellX, cellY), vec4<f32>(vec3<f32>(0.009 + agent.intensity), 1.0));
}