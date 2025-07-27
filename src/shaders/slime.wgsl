fn triple32(x: u32) -> u32 {
    var y = x;
    y = y ^ (y >> 17u);
    y = y * 0xed5ad4bbu;
    y = y ^ (y >> 11u);
    y = y * 0xac4c1b51u;
    y = y ^ (y >> 15u);
    y = y * 0x31848babu;
    y = y ^ (y >> 14u);
    return y;
}

struct Agent {
    // position: vec2<f32>;
	posX: f32,
    posY: f32,
    angle: f32
	//intensity: f32;
};
struct Agents {
    data: array<Agent>
};

struct SpeciesSettings {
    moveSpeed: f32,
    turnSpeed: f32,
    sensorAngleDegrees: f32,
    sensorOffsetDst: f32,
    sensorSize: f32,
    colourR: f32,
    colourG: f32,
    colourB: f32,
    colourA: f32
};

struct ShaderParams {
    numAgents: f32,
    width: f32,
    height: f32,
    delta: f32,
    time: f32
};
@group(0) @binding(0)
var<uniform> shaderParams : ShaderParams;

@group(0) @binding(1)
var<uniform> speciesSettings: SpeciesSettings;

@group(0) @binding(2)
var<storage, read_write> agents: Agents;
struct FloatArray {
    elements: array<f32>,
};
// @group(0) @binding(3) var<storage, read_write> Texture : FloatArray;
@group(0) @binding(3) var SourceTexture : texture_storage_2d<rgba32float, read_write>;



fn sense(agent: Agent, settings: SpeciesSettings, sensorAngleOffset: f32) -> f32 {
    let sensorAngle = agent.angle + sensorAngleOffset;
    let sensorDir = vec2<f32>(cos(sensorAngle), sin(sensorAngle));
    let position = vec2<f32>(agent.posX, agent.posY);
    let sensorPos = position + sensorDir * settings.sensorOffsetDst;
    let sensorCentreX = i32(sensorPos.x);
    let sensorCentreY = i32(sensorPos.y);

    var sum = 0.0;

    for (var offsetX = -i32(settings.sensorSize); offsetX <= i32(settings.sensorSize); offsetX++) {
        for (var offsetY = -i32(settings.sensorSize); offsetY <= i32(settings.sensorSize); offsetY++) {
			// let sampleX = min(i32(shaderParams.width) - 1, max(0, sensorCentreX + i32(offsetX)));
			// let sampleY = min(i32(shaderParams.height) - 1, max(0, sensorCentreY + i32(offsetY)));
            let sampleX = sensorCentreX + i32(offsetX);
            let sampleY = sensorCentreY + i32(offsetY);
			//let offset : i32 = sampleY * i32(shaderParams.width) * 4 + sampleX * 4;
            // sum = sum + dot(vec4<f32>(1.0), vec4<f32>(
            //     textureLoad(SourceTexture, vec2<i32>(sampleX, sampleY)).r,
            // ));
            sum = sum + textureLoad(SourceTexture, vec2<i32>(sampleX, sampleY)).r;
        }
    }

    return sum;
}

fn scaleToRange01(state: u32) -> f32 {
    return f32(state) / 4294967295.0;
}

const PI_OVER_180 : f32 = 0.01745329251;
const TWO_PI : f32 = 6.28318530718;


@compute @workgroup_size(128,1,1)
fn update(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= u32(shaderParams.numAgents) {
        return;
    }

    var agent = agents.data[id.x];
    let pos = vec2<f32>(agent.posX, agent.posY);

	//let intPos = vec2<i32>(i32(pos.x), i32(pos.y));
	//let oldIntensity = textureLoad(SourceTexture, intPos).b;
	//textureStore(SourceTexture, intPos, vec4<f32>(oldIntensity, oldIntensity, 0.0, 1.0));

    var random = triple32(u32(pos.y * f32(shaderParams.width) + pos.x) + triple32(id.x + u32(shaderParams.time * 100000.0)));

	// Steer based on sensory data
    var sensorAngleRad = speciesSettings.sensorAngleDegrees * PI_OVER_180;
    var weightForward = sense(agent, speciesSettings, 0.0);
    var weightLeft = sense(agent, speciesSettings, sensorAngleRad);
    var weightRight = sense(agent, speciesSettings, -sensorAngleRad);


    var randomSteerStrength = scaleToRange01(random);
    var turnSpeed = speciesSettings.turnSpeed * TWO_PI;

	// Continue in same direction
	//if (weightForward > weightLeft && weightForward > weightRight) {
		// agents[id.x].angle += 0
    
	//
    let shouldTurnRandomly = clamp((sign(weightLeft - weightForward) + sign(weightRight - weightForward)) / 2.0, 0.0, 1.0);
    let shouldTurnNormally = abs((sign(weightForward - weightLeft) - sign(weightForward - weightRight)) / 2.0);
	//if (weightForward < weightLeft && weightForward < weightRight) {
    agents.data[id.x].angle = agents.data[id.x].angle + (((randomSteerStrength - 0.5) * 2.0 * turnSpeed * shaderParams.delta) * shouldTurnRandomly);
	//}

	// Turn right
	//elseif (weightRight > weightLeft) {
    agents.data[id.x].angle = agents.data[id.x].angle + (shouldTurnNormally * sign(weightLeft - weightRight) * (randomSteerStrength * turnSpeed * shaderParams.delta));
	//}
	// Turn left
	//elseif (weightLeft > weightRight) {
	//	agents.data[id.x].angle = agents.data[id.x].angle + (randomSteerStrength * turnSpeed * shaderParams.delta);
	//}


	// Update position
    var direction = vec2<f32>(cos(agent.angle), sin(agent.angle));
    var newPos: vec2<f32> = pos + direction * shaderParams.delta * speciesSettings.moveSpeed;

	
	// Clamp position to map boundaries, and pick new random move dir if hit boundary
	//if (newPos.x < 0.0 || newPos.x >= f32(shaderParams.width) || newPos.y < 0.0 || newPos.y >= f32(shaderParams.height)) {
	//	random = triple32(random);
	//	var randomAngle = scaleToRange01(random) * TWO_PI;
//
//		newPos.x = min(f32(shaderParams.width - 1.0),max(0.0, newPos.x));
//		newPos.y = min(f32(shaderParams.height - 1.0),max(0.0, newPos.y));
//		agents.data[id.x].angle = randomAngle;
//	}
	// else {
	//     // var offset : i32 = i32(newPos.y) * i32(shaderParams.width) * 4 + i32(newPos.x) * 4;
	// 	// var oldTrail : vec4<f32> = vec4<f32>(TrailMap.elements[offset], TrailMap.elements[offset + 1], TrailMap.elements[offset + 2], TrailMap.elements[offset + 3]);
    //     // var newVal : vec4<f32> = min(vec4<f32>(1., 1., 1., 1.), oldTrail + vec4<f32>(1.0) * vec4<f32>(shaderParams.delta * shaderParams.delta, shaderParams.delta * shaderParams.delta, shaderParams.delta * shaderParams.delta, shaderParams.delta * shaderParams.delta));
	// 	// TrailMap.elements[offset] = newVal.x;
	// 	// TrailMap.elements[offset + 1] = newVal.y;
	// 	// TrailMap.elements[offset + 2] = newVal.z;
	// 	// TrailMap.elements[offset + 3] = newVal.w;
	// }
    agents.data[id.x].posX = newPos.x;
    agents.data[id.x].posY = newPos.y;
    let intNewPos = vec2<i32>(i32(newPos.x), i32(newPos.y));
    let pix = textureLoad(SourceTexture, intNewPos);
    
    // Make trail deposition frame-rate independent using delta time
    let baseTrailIntensity = 0.009;
    let trailIntensity = baseTrailIntensity * shaderParams.delta * 10.0;
    textureStore(SourceTexture, intNewPos, vec4<f32>(pix.r, 0.0, trailIntensity + pix.r, 1.0));
}