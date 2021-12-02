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


fn rgb2hsv(c: vec3<f32>) -> vec3<f32> {
    let K = vec4<f32>(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    let p = mix(vec4<f32>(c.bg, K.wz), vec4<f32>(c.gb, K.xy), step(c.b, c.g));
    let q = mix(vec4<f32>(p.xyw, c.r), vec4<f32>(c.r, p.yzx), step(p.x, c.r));

    let d = q.x - min(q.w, q.y);
    let e = 1.0e-10;
    return vec3<f32>(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}
fn hsv2rgb(c: vec3<f32>) -> vec3<f32> {
    let K = vec4<f32>(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    let p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, vec3<f32>(0.0), vec3<f32>(1.0)), c.y);
}

[[stage(compute), workgroup_size(16,16,1)]]
fn diffuse([[builtin(global_invocation_id)]] id: vec3<u32>) {
    let diffuseRate = 20.0;
    let decayRate = 0.25;
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
	let decayAmount = decayRate * decayRate * shaderParams.deltaTime;
	var hslCol = rgb2hsv(blurredCol.rgb);
	hslCol.b = clamp(hslCol.b / 1.01, 0.0, 1.0);
	hslCol.g = clamp(hslCol.g / 0.9, 0.0, 1.0);
	hslCol.r = clamp(hslCol.r + decayAmount * 2.0, 0.0, 0.638);
	let decayedCol = hsv2rgb(hslCol);
	//DiffusedTrailMap[id.xy] = blurredCol * saturate(1 - decayRate * deltaTime);
	textureStore(PongTexture, vec2<i32>(id.xy), max(vec4<f32>(0.0), vec4<f32>(decayedCol, 1.0)));
}

