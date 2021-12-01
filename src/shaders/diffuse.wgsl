[[block]]struct ShaderParams {
    numAgents: f32;
    width: f32;
    height: f32;
    trailWeight: f32;
    deltaTime: f32;
    time: f32;
};

[[group(0), binding(0)]] var<uniform> shaderParams : ShaderParams;
[[group(0), binding(1)]] var PingTexture : texture_storage_2d<rgba32float, read>;
[[group(0), binding(2)]] var PongTexture : texture_storage_2d<rgba32float, write>;

[[stage(compute), workgroup_size(16,16,1)]]
fn diffuse([[builtin(global_invocation_id)]] id: vec3<u32>)
{
    let diffuseRate = 0.8;
    let decayRate = 0.07;
	if (id.x < 0u || id.x >= u32(shaderParams.width) || id.y < 0u || id.y >= u32(shaderParams.height)) {
		return;
	}

	var sum = vec4<f32>(0.0);
	var originalCol = textureLoad(PingTexture, vec2<i32>(id.xy));
	// 3x3 blur
	for (var offsetX = -1; offsetX <= 1; offsetX = offsetX + 1) {
		for (var offsetY = -1; offsetY <= 1; offsetY = offsetY + 1) {
			var sampleX = min(i32(shaderParams.width) - 1, max(0, i32(id.x) + offsetX));
			var sampleY = min(i32(shaderParams.height) - 1, max(0, i32(id.y) + offsetY));
			sum = sum + textureLoad(PingTexture, vec2<i32>(sampleX,sampleY));
		}
	}

	var blurredCol = sum / vec4<f32>(9.0);
	var diffuseWeight = clamp(diffuseRate * shaderParams.deltaTime, 0.0, 1.0); // saturate()
	blurredCol = originalCol * (1.0 - diffuseWeight) + blurredCol * (diffuseWeight);

	//DiffusedTrailMap[id.xy] = blurredCol * saturate(1 - decayRate * deltaTime);
	textureStore(PongTexture, vec2<i32>(id.xy), max(vec4<f32>(0.0), blurredCol - decayRate * decayRate * shaderParams.deltaTime));
}