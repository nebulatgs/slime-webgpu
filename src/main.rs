use std::{sync::Arc, time::Instant};

use clap::{arg, command, Parser};
use rand::Rng;
use wgpu::{util::DeviceExt, BindGroup, BufferAddress, BufferDescriptor, BufferUsages, Device};
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Fullscreen, Window, WindowId},
};
static AGENTS_PER_GROUP: u32 = 128;
static NUM_AGENTS: u32 = (1 << 23) - AGENTS_PER_GROUP;
static DIFFUSE_TILE_SIZE: u32 = 16;
static SCALE_DOWN_FACTOR: f32 = 1.0;
static SIM_WIDTH: u32 = (3840.0 * SCALE_DOWN_FACTOR) as _;
static SIM_HEIGHT: u32 = (2160.0 * SCALE_DOWN_FACTOR) as _;

/// Slime Simulation
#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    /// Enable VSync
    #[arg(long)]
    vsync: bool,
}

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
    delta: f32,
    time: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[allow(non_snake_case)]
struct RenderParams {
    width: f32,
    height: f32,
    scaleDownFactor: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[allow(non_snake_case)]
struct Agent {
    posX: f32,
    posY: f32,
    angle: f32,
    // intensity: f32,
}
struct State<'a> {
    surface: wgpu::Surface<'a>,
    window: Arc<Box<winit::window::Window>>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    clear_color: wgpu::Color,
    render_pipeline: wgpu::RenderPipeline,
    render_bind_group: BindGroup,
    compute_pipeline: wgpu::ComputePipeline,
    compute_bind_group: BindGroup,
    compute_diffuse_pipeline: wgpu::ComputePipeline,
    compute_diffuse_bind_group: BindGroup,
    shader_param_buffer: wgpu::Buffer,
    species_param_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    time: f32,
    ping_texture: wgpu::Texture,
    pong_texture: wgpu::Texture,
    then: Instant,
    bundle: wgpu::RenderBundle,
    shader_param_data: ShaderParams,
    species_param_data: SpeciesSettings,

    sim_texture_view: wgpu::TextureView,
    scaling_pipeline: wgpu::RenderPipeline,
    scaled_texture_bind_group: wgpu::BindGroup,
    sampler: wgpu::Sampler,
    uniform_buffer: wgpu::Buffer,
    args: Args,
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

impl<'a> State<'a> {
    // Creating some of the wgpu types requires async code
    async fn new(window: Window, args: Args) -> Self {
        let size = window.inner_size();
        // window.set_fullscreen(Some(Fullscreen::Exclusive(
        //     window
        //         .primary_monitor()
        //         .unwrap()
        //         .video_modes()
        //         .next()
        //         .unwrap(),
        // )));
        window.set_fullscreen(Some(Fullscreen::Borderless(None)));
        // window.set_fullscreen(None);
        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let window = Arc::new(Box::new(window));
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                label: None,
            })
            .await
            .unwrap();

        let vsync_mode = if args.vsync {
            wgpu::PresentMode::AutoVsync
        } else {
            wgpu::PresentMode::AutoNoVsync
        };

        let config = wgpu::SurfaceConfiguration {
            desired_maximum_frame_latency: 2,
            view_formats: vec![surface.get_capabilities(&adapter).formats[0]],
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_capabilities(&adapter).formats[0],
            width: size.width,
            height: size.height,
            present_mode: vsync_mode,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
        };
        surface.configure(&device, &config);

        let clear_color = wgpu::Color::BLACK;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
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
            view_formats: &[wgpu::TextureFormat::Rgba32Float],
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
            view_formats: &[wgpu::TextureFormat::Rgba32Float],
        });
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let render_param_data = RenderParams {
            width: SIM_WIDTH as _,
            height: SIM_HEIGHT as _,
            scaleDownFactor: SCALE_DOWN_FACTOR as _,
        };
        let render_param_slice = &[render_param_data];
        let render_param_slice: &[u8] = bytemuck::cast_slice(render_param_slice);
        let render_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    // Texture Sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
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
                    // Render Params Buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(render_param_slice.len() as _),
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
            cache: None,
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            multiview: None,
            vertex: wgpu::VertexState {
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                module: &shader,
                entry_point: Some("vs_main"), // 1.
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3],
                }], // 2.
            },
            fragment: Some(wgpu::FragmentState {
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                // 3.
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    // 4.
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList, // 1.
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw, // 2.
                cull_mode: Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLAMPING
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: None, // 1.
            multisample: wgpu::MultisampleState::default(), // multisample: wgpu::MultisampleState {
                                 //     count: SAMPLE_COUNT,              // 2.
                                 //     mask: !0,                         // 3.
                                 //     alpha_to_coverage_enabled: false, // 4.
                                 // },
        });
        let render_param_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Render Parameter Buffer"),
            contents: bytemuck::cast_slice(&[render_param_data]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, // | wgpu::BufferUsages::MAP_WRITE,
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
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: render_param_buffer.as_entire_binding(),
                },
            ],
            label: None,
        });

        let mut encoder =
            device.create_render_bundle_encoder(&wgpu::RenderBundleEncoderDescriptor {
                label: None,
                multiview: None,
                color_formats: &[Some(config.format)],
                depth_stencil: None,
                sample_count: 1,
            });
        encoder.set_pipeline(&render_pipeline);
        encoder.set_vertex_buffer(0, vertex_buffer.slice(..));
        encoder.draw(0..VERTICES.len() as _, 0..1);
        let bundle = encoder.finish(&wgpu::RenderBundleDescriptor {
            label: Some("main"),
        });

        let shader_param_data = ShaderParams {
            numAgents: NUM_AGENTS as _,
            width: SIM_WIDTH as _,
            height: SIM_HEIGHT as _,
            delta: 0.03,
            time: 0.0,
        };
        let shader_param_slice = &[shader_param_data];
        let shader_param_slice: &[u8] = bytemuck::cast_slice(shader_param_slice);

        let shader_param_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Shader Parameter Buffer"),
            contents: shader_param_slice,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, // | wgpu::BufferUsages::MAP_WRITE,
        });
        let species_param_data = SpeciesSettings {
            moveSpeed: 120.0,
            turnSpeed: -4.0,
            sensorAngleDegrees: 112.0,
            sensorOffsetDst: 50.0,
            sensorSize: 0.0,
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
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, // | wgpu::BufferUsages::MAP_WRITE,
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
                            access: wgpu::StorageTextureAccess::ReadWrite,
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
        let compute_diffuse_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("compute diffuse"),
                bind_group_layouts: &[&compute_diffuse_bind_group_layout],
                push_constant_ranges: &[],
            });
        // // Compute shader pipeline
        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/slime.wgsl").into()),
        });
        let compute_diffuse_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute Diffuse Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/diffuse.wgsl").into()),
        });
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            label: Some("Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: Some("update"),
        });
        let compute_diffuse_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                cache: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                label: Some("Compute Diffuse Pipeline"),
                layout: Some(&compute_diffuse_pipeline_layout),
                module: &compute_diffuse_shader,
                entry_point: Some("diffuse"),
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
        // Create fixed-size simulation texture with matching format
        let sim_texture = device.create_texture(&wgpu::TextureDescriptor {
            view_formats: &[config.format],
            size: wgpu::Extent3d {
                width: SIM_WIDTH,
                height: SIM_HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: config.format, // Use same format as surface config
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            label: Some("simulation_texture"),
        });

        let sim_texture_view = sim_texture.create_view(&Default::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scaling Uniforms"),
            size: std::mem::size_of::<[[f32; 4]; 4]>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group layout and bind group
        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("texture_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let scaled_texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("texture_bind_group"),
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&sim_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        // Create pipeline layout
        let scaling_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("scaling_pipeline_layout"),
                bind_group_layouts: &[&texture_bind_group_layout],
                push_constant_ranges: &[],
            });

        // Create scaling shader
        let scaling_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scaling_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/scaling.wgsl").into()),
        });

        // Create scaling pipeline
        let scaling_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            cache: None,
            label: Some("scaling_pipeline"),
            layout: Some(&scaling_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &scaling_shader,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3],
                }], // 2.  // Add the vertex buffer layout here
            },
            fragment: Some(wgpu::FragmentState {
                module: &scaling_shader,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        Self {
            args,
            window,
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
            compute_diffuse_pipeline,
            compute_diffuse_bind_group,
            vertex_buffer,
            shader_param_buffer,
            species_param_buffer,
            time: shader_param_data.time,
            ping_texture,
            pong_texture,
            then: Instant::now(),
            bundle,
            shader_param_data,
            species_param_data,
            sim_texture_view,
            scaling_pipeline,
            scaled_texture_bind_group,
            sampler,
            uniform_buffer,
        }
    }

    fn build_agent_buffer(device: &Device) -> wgpu::Buffer {
        let mut agents = vec![
            Agent {
                posX: 0.0,
                posY: 0.0,
                angle: 0.0,
                // intensity: 0.0,
            };
            NUM_AGENTS as _
        ];

        let mut rng = rand::rng();
        let now = std::time::Instant::now();
        for agent in &mut agents {
            static R: f64 = 300.0;
            static CENTER_X: f64 = SIM_WIDTH as f64 / 2.0;
            static CENTER_Y: f64 = SIM_HEIGHT as f64 / 2.0;
            static RAD_TO_DEG: f64 = 180.0 * std::f64::consts::FRAC_1_PI;

            let r = R * rng.random_range::<f64, _>(0.0..1.0).sqrt();
            let theta = rng.random_range::<f64, _>(0.0..1.0) * 2.0 * std::f64::consts::PI;
            agent.posX = (CENTER_X + r * theta.cos()) as f32;
            agent.posY = (CENTER_Y + r * theta.sin()) as f32;
            let angle = rng.random_range::<f64, _>(0.0..1.0) * 2.0 * std::f64::consts::PI;
            // agent.posX = 100.0;
            // agent.posY = 100.0;
            agent.angle = ((angle + std::f64::consts::PI) * RAD_TO_DEG) as f32;
            // agent.posX = rng.random_range(0.0..SIM_WIDTH as f32/ 2.0);
            // agent.posY = rng.random_range(-(SIM_HEIGHT  as f32/ 2.0)..SIM_HEIGHT as f32);
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

            let projection =
                get_projection_matrix(new_size.width as f32, new_size.height as f32, true);

            self.queue
                .write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&projection));
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                device_id: _,
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::KeyD),
                        ..
                    },
                ..
            } => {
                self.species_param_data.moveSpeed += 1.0;
                self.update_uniform_buffer(&self.species_param_buffer, &self.species_param_data);
                true
            }
            WindowEvent::KeyboardInput {
                device_id: _,
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::KeyL),
                        ..
                    },
                ..
            } => {
                std::thread::sleep(std::time::Duration::from_millis(20));
                true
            }
            WindowEvent::KeyboardInput {
                device_id: _,
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::KeyA),
                        ..
                    },
                ..
            } => {
                self.species_param_data.moveSpeed =
                    (self.species_param_data.moveSpeed - 1.0).max(0.0);
                self.update_uniform_buffer(&self.species_param_buffer, &self.species_param_data);
                true
            }

            WindowEvent::KeyboardInput {
                device_id: _,
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::KeyQ),
                        ..
                    },
                ..
            } => {
                self.species_param_data.turnSpeed =
                    (self.species_param_data.turnSpeed + 1.0).min(0.0);
                self.update_uniform_buffer(&self.species_param_buffer, &self.species_param_data);
                true
            }
            WindowEvent::KeyboardInput {
                device_id: _,
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::KeyE),
                        ..
                    },
                ..
            } => {
                self.species_param_data.turnSpeed -= 1.0;
                self.update_uniform_buffer(&self.species_param_buffer, &self.species_param_data);
                true
            }
            WindowEvent::KeyboardInput {
                device_id: _,
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::KeyS),
                        ..
                    },
                ..
            } => {
                self.species_param_data.sensorOffsetDst =
                    (self.species_param_data.sensorOffsetDst - 1.0).max(0.0);
                self.update_uniform_buffer(&self.species_param_buffer, &self.species_param_data);
                true
            }
            WindowEvent::KeyboardInput {
                device_id: _,
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::KeyW),
                        ..
                    },
                ..
            } => {
                self.species_param_data.sensorOffsetDst += 1.0;
                self.update_uniform_buffer(&self.species_param_buffer, &self.species_param_data);
                true
            }
            _ => false,
        }
    }

    fn update_uniform_buffer<T: bytemuck::Pod + Send + Sync + std::fmt::Debug>(
        &self,
        buffer: &wgpu::Buffer,
        data: &T,
    ) {
        let bytes = bytemuck::bytes_of(data);
        // dbg!(&data);
        // self.queue.write_buffer(buffer, 0, bytemuck::bytes_of(data));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Command Encoder"),
            });
        let device_buffer = self.device.create_buffer(&BufferDescriptor {
            label: None,
            size: bytes.len() as u64,
            usage: BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });
        let buffer_slice = device_buffer.slice(..);
        let _ = self.device.poll(wgpu::wgt::PollType::Wait);
        buffer_slice.get_mapped_range_mut()[..bytes.len()].copy_from_slice(bytes);
        device_buffer.unmap();
        encoder.copy_buffer_to_buffer(
            &device_buffer,
            0,
            &buffer,
            0,
            std::mem::size_of_val(&bytes) as BufferAddress,
        ); //A mutable borrow to the command encoder is needed here in order to update uniforms
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    fn update(&mut self) {
        let now = Instant::now();
        let delta = now.duration_since(self.then).as_secs_f32();
        self.then = now;

        // Time is used for shader RNG
        self.time = self.time + delta;

        let shader_param_data = ShaderParams {
            numAgents: NUM_AGENTS as _,
            width: SIM_WIDTH as _,
            height: SIM_HEIGHT as _,
            delta,
            time: self.time,
        };
        println!("delta: {}", shader_param_data.delta);
        self.update_uniform_buffer(&self.shader_param_buffer, &shader_param_data);
    }

    fn draw(&mut self) -> Result<(), wgpu::SurfaceError> {
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
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.sim_texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            let mut encoder =
                self.device
                    .create_render_bundle_encoder(&wgpu::RenderBundleEncoderDescriptor {
                        label: None,
                        multiview: None,
                        color_formats: &[Some(self.config.format)],
                        depth_stencil: None,
                        sample_count: 1,
                    });
            encoder.set_pipeline(&self.render_pipeline);
            encoder.set_bind_group(0, &self.render_bind_group, &[]);
            encoder.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            encoder.draw(0..VERTICES.len() as _, 0..1);
            self.bundle = encoder.finish(&wgpu::RenderBundleDescriptor {
                label: Some("main"),
            });
            render_pass.execute_bundles(std::iter::once(&self.bundle));
        }

        {
            let mut scaling_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scaling_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });

            scaling_pass.set_pipeline(&self.scaling_pipeline);
            scaling_pass.set_bind_group(0, &self.scaled_texture_bind_group, &[]);
            scaling_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            scaling_pass.draw(0..VERTICES.len() as _, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Command Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                ..Default::default()
            });
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);
            compute_pass.dispatch_workgroups(NUM_AGENTS / AGENTS_PER_GROUP, 1, 1);
        }
        {
            let mut compute_diffuse_pass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Compute Diffuse Pass"),
                    ..Default::default()
                });
            compute_diffuse_pass.set_pipeline(&self.compute_diffuse_pipeline);
            compute_diffuse_pass.set_bind_group(0, &self.compute_diffuse_bind_group, &[]);
            compute_diffuse_pass.dispatch_workgroups(
                SIM_WIDTH / DIFFUSE_TILE_SIZE,
                SIM_HEIGHT / DIFFUSE_TILE_SIZE,
                1,
            );
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

        Ok(())
    }
}
fn get_projection_matrix(
    window_width: f32,
    window_height: f32,
    center_crop: bool,
) -> [[f32; 4]; 4] {
    let window_aspect = window_width / window_height;
    let sim_aspect = (SIM_WIDTH as f32) / (SIM_HEIGHT as f32);

    let (width, height) = if window_aspect > sim_aspect {
        if center_crop {
            // Window is wider - scale up sim width to match window width
            let width = 1.0;
            let height = sim_aspect / window_aspect;
            (width, height)
        } else {
            // Window is wider - letterbox width
            let height = 1.0;
            let width = window_aspect / sim_aspect;
            (width, height)
        }
    } else {
        if center_crop {
            // Window is taller - scale up sim height to match window height
            let width = window_aspect / sim_aspect;
            let height = 1.0;
            (width, height)
        } else {
            // Window is taller - letterbox height
            let width = 1.0;
            let height = sim_aspect / window_aspect;
            (width, height)
        }
    };

    cgmath::ortho(-width, width, -height, height, -1.0, 1.0).into()
}

#[derive(Default)]
struct SlimeSim<'a> {
    // window: Option<Window>,
    args: Option<Args>,
    state: Option<State<'a>>,
}

impl<'a> ApplicationHandler for SlimeSim<'a> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes = Window::default_attributes().with_title("Slime");
        let window = event_loop.create_window(window_attributes).unwrap();
        let mut state =
            pollster::block_on(State::<'static>::new(window, self.args.take().unwrap()));
        state.resize(state.size);
        // state.window.request_redraw();
        self.state = Some(state);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let state = self.state.as_mut().unwrap();
        state.input(&event);
        match event {
            WindowEvent::RedrawRequested => {
                // state.then = Instant::now();
                match state.draw() {
                    Ok(_) => {}
                    // Reconfigure the surface if lost
                    Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                    // The system is out of memory, we should probably quit
                    Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                    // All other errors (Outdated, Timeout) should be resolved by the next frame
                    Err(e) => eprintln!("{:?}", e),
                }
                state.update();
                match state.render() {
                    Ok(_) => {}
                    // Reconfigure the surface if lost
                    Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                    // The system is out of memory, we should probably quit
                    Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                    // All other errors (Outdated, Timeout) should be resolved by the next frame
                    Err(e) => eprintln!("{:?}", e),
                }
                state.window.request_redraw();
            }

            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        ..
                    },
                ..
            } => event_loop.exit(),
            WindowEvent::Resized(physical_size) => {
                state.resize(physical_size);
            }
            // WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
            //     state.resize(**new_inner_size);
            // }
            _ => {}
        }
    }
}
fn main() {
    let args = Args::parse();
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    let mut app = SlimeSim {
        args: Some(args),
        state: None,
    };

    event_loop.run_app(&mut app).unwrap();
}
