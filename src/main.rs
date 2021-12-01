use std::{mem, time::Instant};

use rand::Rng;
use wgpu::{util::DeviceExt, BindGroup, Device, MapMode};
use winit::{
    dpi::PhysicalSize,
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder, Fullscreen}, monitor::VideoMode,
};
const NUM_AGENTS: u32 = 500_000;
const AGENTS_PER_GROUP: u32 = 256;
const AGENTS_PER_DRAW_GROUP: u32 = 16;
const SIM_WIDTH: u32 = 1920;
const SIM_HEIGHT: u32 = 1080;
const SCALE_DOWN_FACTOR: u32 = 2;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[allow(non_snake_case)]
struct SpeciesSettings {
    moveSpeed: f32,
    turnSpeed: f32,
    sensorAngleDegrees: f32,
    sensorOffsetDst: f32,
    sensorSize: f32,
    colourR: f32,
    colourG: f32,
    colourB: f32,
    colourA: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[allow(non_snake_case)]
struct ShaderParams {
    numAgents: f32,
    width: f32,
    height: f32,
    trailWeight: f32,
    deltaTime: f32,
    time: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[allow(non_snake_case)]
struct Agent {
    posX: f32,
    posY: f32,
    angle: f32,
}
struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    clear_color: wgpu::Color,
    render_pipeline: wgpu::RenderPipeline,
    render_bind_group: BindGroup,
    compute_pipeline: wgpu::ComputePipeline,
    compute_bind_group: BindGroup,
    compute_draw_pipeline: wgpu::ComputePipeline,
    compute_draw_bind_group: BindGroup,
    compute_diffuse_pipeline: wgpu::ComputePipeline,
    compute_diffuse_bind_group: BindGroup,
    shader_param_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    time: f32,
    ping_texture: wgpu::Texture,
    pong_texture: wgpu::Texture,
    then: Instant,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
}

const VERTICES: &[Vertex] = &[
    Vertex {
        position: [-1.0, 1.0, 0.0],
    },
    Vertex {
        position: [-1.0, -1.0, 0.0],
    },
    Vertex {
        position: [1.0, -1.0, 0.0],
    },
    Vertex {
        position: [1.0, -1.0, 0.0],
    },
    Vertex {
        position: [1.0, 1.0, 0.0],
    },
    Vertex {
        position: [-1.0, 1.0, 0.0],
    },
];

impl State {
    // Creating some of the wgpu types requires async code
    async fn new(window: &Window) -> Self {
        let size = window.inner_size();

        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let surface = unsafe { instance.create_surface(window) };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
                    limits: wgpu::Limits::default(),
                    label: None,
                },
                None, // Trace path
            )
            .await
            .unwrap();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_preferred_format(&adapter).unwrap(),
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };
        surface.configure(&device, &config);
        let clear_color = wgpu::Color::BLACK;

        let shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("Render Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/shader.wgsl").into()),
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());
        let ping_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Ping Texture"),
            size: wgpu::Extent3d {
                width: SIM_WIDTH,
                height: SIM_HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
        });
        let pong_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Pong Texture"),
            size: wgpu::Extent3d {
                width: SIM_WIDTH,
                height: SIM_HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
        });
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let render_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    // Texture Sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler {
                            comparison: false,
                            filtering: false,
                        },
                        count: None,
                    },
                    // Storage Texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::ReadOnly,
                            format: wgpu::TextureFormat::Rgba32Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
                label: None,
            });
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&render_bind_group_layout],
                push_constant_ranges: &[],
            });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main", // 1.
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3],
                }], // 2.
            },
            fragment: Some(wgpu::FragmentState {
                // 3.
                module: &shader,
                entry_point: "fs_main",
                targets: &[wgpu::ColorTargetState {
                    // 4.
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                }],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList, // 1.
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw, // 2.
                cull_mode: Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLAMPING
                clamp_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: None, // 1.
            multisample: wgpu::MultisampleState {
                count: 1,                         // 2.
                mask: !0,                         // 3.
                alpha_to_coverage_enabled: false, // 4.
            },
        });
        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &render_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        &ping_texture.create_view(&wgpu::TextureViewDescriptor::default()),
                    ),
                },
            ],
            label: None,
        });
        let shader_param_data = ShaderParams {
            numAgents: NUM_AGENTS as _,
            width: SIM_WIDTH as _,
            height: SIM_HEIGHT as _,
            trailWeight: 0.5,
            deltaTime: 0.0,
            time: 0.0,
        };
        let shader_param_slice = &[shader_param_data];
        let shader_param_slice: &[u8] = bytemuck::cast_slice(shader_param_slice);

        let shader_param_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Shader Parameter Buffer"),
            contents: bytemuck::cast_slice(&[shader_param_data]),
            usage: wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::MAP_WRITE,
        });
        let species_param_data = SpeciesSettings {
            moveSpeed: 50.0,
            turnSpeed: -6.0,
            sensorAngleDegrees: 112.0,
            sensorOffsetDst: 50.0,
            sensorSize: 1.0,
            colourR: 0.0,
            colourG: 1.0,
            colourB: 0.0,
            colourA: 1.0,
        };
        let species_param_slice = &[species_param_data];
        let species_param_slice: &[u8] = bytemuck::cast_slice(species_param_slice);

        let species_param_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Species Parameter Buffer"),
            contents: species_param_slice,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    // Shader Parameter Buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(shader_param_slice.len() as _),
                        },
                        count: None,
                    },
                    // Species Parameter Buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(species_param_slice.len() as _),
                        },
                        count: None,
                    },
                    // Agents Buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new((NUM_AGENTS * 4 * 3) as _),
                        },
                        count: None,
                    },
                    // Storage Texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::ReadOnly,
                            format: wgpu::TextureFormat::Rgba32Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
                label: None,
            });
        let compute_draw_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    // Shader Parameter Buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(shader_param_slice.len() as _),
                        },
                        count: None,
                    },
                    // Agents Buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new((NUM_AGENTS * 4 * 3) as _),
                        },
                        count: None,
                    },
                    // Texture Sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler {
                            comparison: false,
                            filtering: false,
                        },
                        count: None,
                    },
                    // Storage Texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba32Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
                label: None,
            });
        let compute_diffuse_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    // Shader Parameter Buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(shader_param_slice.len() as _),
                        },
                        count: None,
                    },
                    // Ping (Read) Texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::ReadOnly,
                            format: wgpu::TextureFormat::Rgba32Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // Pong (Write) Texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba32Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
                label: None,
            });
        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("compute"),
                bind_group_layouts: &[&compute_bind_group_layout],
                push_constant_ranges: &[],
            });
        let compute_draw_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("compute draw"),
                bind_group_layouts: &[&compute_draw_bind_group_layout],
                push_constant_ranges: &[],
            });
        let compute_diffuse_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("compute diffuse"),
                bind_group_layouts: &[&compute_diffuse_bind_group_layout],
                push_constant_ranges: &[],
            });
        // // Compute shader pipeline
        let compute_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/slime.wgsl").into()),
        });
        let compute_draw_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("Compute Draw Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/draw.wgsl").into()),
        });
        let compute_diffuse_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("Compute Diffuse Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/diffuse.wgsl").into()),
        });
        // let compute_pipeline_layout =
        //     device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        //         label: Some("Compute Pipeline Layout"),
        //         bind_group_layouts: &[],
        //         push_constant_ranges: &[],
        //     });
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: "update",
        });
        let compute_draw_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Compute Draw Pipeline"),
                layout: Some(&compute_draw_pipeline_layout),
                module: &compute_draw_shader,
                entry_point: "main",
            });
        let compute_diffuse_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Compute Diffuse Pipeline"),
                layout: Some(&compute_diffuse_pipeline_layout),
                module: &compute_diffuse_shader,
                entry_point: "diffuse",
            });
        let agent_buffer = Self::build_agent_buffer(&device);
        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &compute_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: shader_param_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: species_param_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: agent_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(
                        &ping_texture.create_view(&wgpu::TextureViewDescriptor::default()),
                    ),
                },
            ],
            label: None,
        });

        let compute_draw_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &compute_draw_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: shader_param_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: agent_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(
                        &ping_texture.create_view(&wgpu::TextureViewDescriptor::default()),
                    ),
                },
            ],
            label: None,
        });
        let compute_diffuse_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &compute_diffuse_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: shader_param_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        &ping_texture.create_view(&wgpu::TextureViewDescriptor::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(
                        &pong_texture.create_view(&wgpu::TextureViewDescriptor::default()),
                    ),
                },
            ],
            label: None,
        });

        Self {
            surface,
            device,
            queue,
            config,
            size,
            clear_color,
            render_pipeline,
            render_bind_group,
            compute_pipeline,
            compute_bind_group,
            compute_draw_pipeline,
            compute_draw_bind_group,
            compute_diffuse_pipeline,
            compute_diffuse_bind_group,
            vertex_buffer,
            shader_param_buffer,
            time: 0.0,
            ping_texture,
            pong_texture,
            then: Instant::now(),
        }
    }

    fn build_agent_buffer(device: &Device) -> wgpu::Buffer {
        let mut agents = vec![
            Agent {
                posX: 0.0,
                posY: 0.0,
                angle: 0.0
            };
            NUM_AGENTS as _
        ];

        let mut rng = rand::thread_rng();
        let now = std::time::Instant::now();
        for agent in &mut agents {
            static R: f32 = 200.0;
            static CENTER_X: f32 = SIM_WIDTH as f32 / 2.0;
            static CENTER_Y: f32 = SIM_HEIGHT as f32 / 2.0;
            static RAD_TO_DEG: f32 = 180.0 / std::f32::consts::PI;

            let r = R * rng.gen_range::<f32, _>(0.0..1.0).sqrt();
            let theta = rng.gen_range::<f32, _>(0.0..1.0) * 2.0 * std::f32::consts::PI;
            agent.posX = CENTER_X + r * theta.cos();
            agent.posY = CENTER_Y + r * theta.sin();
            agent.angle = (theta * RAD_TO_DEG); 
            // agent.posX = rng.gen_range(0.0..SIM_WIDTH as f32/ 2.0);
            // agent.posY = rng.gen_range(-(SIM_HEIGHT  as f32/ 2.0)..SIM_HEIGHT as f32);
        }

        println!("generated agents in {}ms", now.elapsed().as_millis());

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Agent Buffer"),
            contents: bytemuck::cast_slice(&agents),
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
        });

        buffer
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::CursorMoved {
                device_id,
                position,
                ..
            } => {
                self.clear_color = wgpu::Color {
                    r: position.x as f64 / self.size.width as f64,
                    g: position.y as f64 / self.size.height as f64,
                    b: 1.0,
                    a: 1.0,
                };
                true
            }
            _ => false,
        }
    }

    fn update(&mut self) {
        self.time = self.time + 0.1;
        let shader_param_data = ShaderParams {
            numAgents: NUM_AGENTS as _,
            width: SIM_WIDTH as _,
            height: SIM_HEIGHT as _,
            trailWeight: 0.5,
            deltaTime: Instant::now()
                .checked_duration_since(self.then)
                .unwrap()
                .as_secs_f32(),
            time: self.time,
        };
        let shader_param_slice = &[shader_param_data];
        let shader_param_slice: &[u8] = bytemuck::cast_slice(shader_param_slice);

        let buffer_slice = self.shader_param_buffer.slice(..);
        let buffer_future = buffer_slice.map_async(wgpu::MapMode::Write);
        self.device.poll(wgpu::Maintain::Wait);
        pollster::block_on(buffer_future).unwrap();
        buffer_slice.get_mapped_range_mut()[..shader_param_slice.len()]
            .copy_from_slice(shader_param_slice);
        self.shader_param_buffer.unmap();
        self.then = Instant::now();
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
            });
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);
            compute_pass.dispatch(NUM_AGENTS / AGENTS_PER_GROUP, 1, 1);
        }
        {
            let mut compute_draw_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Draw Pass"),
            });
            compute_draw_pass.set_pipeline(&self.compute_draw_pipeline);
            compute_draw_pass.set_bind_group(0, &self.compute_draw_bind_group, &[]);
            compute_draw_pass.dispatch(NUM_AGENTS / AGENTS_PER_DRAW_GROUP, 1, 1);
        }
        {
            let mut compute_diffuse_pass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Compute Diffuse Pass"),
                });
            compute_diffuse_pass.set_pipeline(&self.compute_diffuse_pipeline);
            compute_diffuse_pass.set_bind_group(0, &self.compute_diffuse_bind_group, &[]);
            compute_diffuse_pass.dispatch(SIM_WIDTH / 16, SIM_HEIGHT / 16, 1);
        }
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.clear_color),
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            });
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.render_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.draw(0..VERTICES.len() as _, 0..1);
        }
        {
            encoder.copy_texture_to_texture(
                self.pong_texture.as_image_copy(),
                self.ping_texture.as_image_copy(),
                wgpu::Extent3d {
                    width: SIM_WIDTH,
                    height: SIM_HEIGHT,
                    depth_or_array_layers: 1,
                },
            );
        }
        // submit will accept anything that implements IntoIter
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_inner_size(PhysicalSize {
            height: SIM_HEIGHT,
            width: SIM_WIDTH,
        })
        .build(&event_loop)
        .unwrap();
    window.set_fullscreen(Some(Fullscreen::Borderless(None)));
    let mut state = pollster::block_on(State::new(&window));

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::RedrawRequested(_) => {
                state.update();
                match state.render() {
                    Ok(_) => {}
                    // Reconfigure the surface if lost
                    Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                    // The system is out of memory, we should probably quit
                    Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                    // All other errors (Outdated, Timeout) should be resolved by the next frame
                    Err(e) => eprintln!("{:?}", e),
                }
            }
            Event::MainEventsCleared => {
                // RedrawRequested will only trigger once, unless we manually
                // request it.
                window.request_redraw();
            }
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => {
                if !state.input(event) {
                    // UPDATED!
                    match event {
                        WindowEvent::CloseRequested
                        | WindowEvent::KeyboardInput {
                            input:
                                KeyboardInput {
                                    state: ElementState::Pressed,
                                    virtual_keycode: Some(VirtualKeyCode::Escape),
                                    ..
                                },
                            ..
                        } => *control_flow = ControlFlow::Exit,
                        WindowEvent::Resized(physical_size) => {
                            state.resize(*physical_size);
                        }
                        WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                            state.resize(**new_inner_size);
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    });
}
