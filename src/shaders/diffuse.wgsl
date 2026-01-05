struct ShaderParams {
    numAgents: f32,
    width: f32,
    height: f32,
    delta: f32,
    time: f32
};

@group(0) @binding(0) var<uniform> shaderParams : ShaderParams;
@group(0) @binding(1) var PingTexture : texture_storage_2d<rg16float, read>;
@group(0) @binding(2) var PongTexture : texture_storage_2d<rg16float, write>;

@compute @workgroup_size(8,8,1)
fn diffuse(@builtin(global_invocation_id) id: vec3<u32>) {
    // Cache constants and frequently used values
    const diffuseRate = 10.0;
    const decayRate = 0.25;
    let delta = shaderParams.delta;
    let width = i32(shaderParams.width);
    let height = i32(shaderParams.height);
    let x = i32(id.x);
    let y = i32(id.y);
    
    // Early bounds check
    if id.x >= u32(shaderParams.width) || id.y >= u32(shaderParams.height) {
        return;
    }

    // Get original value (only need blue channel)
    let originalValue = textureLoad(PingTexture, vec2<i32>(x, y)).g;
    
    // 3x3 blur - work with scalars instead of vectors
    var sum = 0.0;
    for (var offsetX = -1; offsetX <= 1; offsetX++) {
        let sampleX = min(width - 1, max(0, x + offsetX));
        for (var offsetY = -1; offsetY <= 1; offsetY++) {
            let sampleY = min(height - 1, max(0, y + offsetY));
            sum += textureLoad(PingTexture, vec2<i32>(sampleX, sampleY)).g;
        }
    }
    
    // Calculate blurred and diffused value
    let blurredValue = sum * (1.0 / 9.0);
    let diffuseWeight = clamp(diffuseRate * delta, 0.0, 1.0);
    let diffusedValue = originalValue + (blurredValue - originalValue) * diffuseWeight;
    
    // Apply exponential decay
    let decayFactor = exp(-decayRate * delta * 5.0);
    let finalValue = max(0.0, diffusedValue * decayFactor);
    
    // Store result (red and blue channels get the same value, green is 0, alpha is 1)
    textureStore(PongTexture, vec2<i32>(x, y), vec4<f32>(finalValue, finalValue, 0.0, 1.0));
}

