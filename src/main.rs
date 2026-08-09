use nannou::prelude::*;
use rand::Rng;
use std::sync::Arc;

const WINDOW_W: u32 = 1280;
const WINDOW_H: u32 = 720;
const NUM_PARTICLES: u32 = 4000;
const NUM_TYPES: u32 = 6;
const WORKGROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Particle {
    pos: [f32; 2],
    vel: [f32; 2],
    ptype: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    half_width: f32,
    half_height: f32,
    dt: f32,
    friction: f32,
    max_radius: f32,
    beta: f32,
    force_scale: f32,
    particle_radius: f32,
    num_types: u32,
    num_particles: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct QuadVertex {
    corner: [f32; 2],
}

const QUAD_VERTICES: [QuadVertex; 6] = [
    QuadVertex { corner: [-1.0, -1.0] },
    QuadVertex { corner: [1.0, -1.0] },
    QuadVertex { corner: [1.0, 1.0] },
    QuadVertex { corner: [-1.0, -1.0] },
    QuadVertex { corner: [1.0, 1.0] },
    QuadVertex { corner: [-1.0, 1.0] },
];

#[derive(Clone)]
struct Model {
    compute_pipeline: Arc<wgpu::ComputePipeline>,
    compute_bind_groups: [Arc<wgpu::BindGroup>; 2],
    render_pipeline: Arc<wgpu::RenderPipeline>,
    render_bind_groups: [Arc<wgpu::BindGroup>; 2],
    quad_vertex_buffer: Arc<wgpu::Buffer>,
    matrix_buffer: Arc<wgpu::Buffer>,
    particle_buffers: [Arc<wgpu::Buffer>; 2],
    current: usize,
    paused: bool,
    params: Params,
}

fn main() {
    nannou::app(model).update(update).render(render).run();
}

fn random_particles(rng: &mut impl Rng, n: u32, num_types: u32, half_w: f32, half_h: f32) -> Vec<Particle> {
    (0..n)
        .map(|_| Particle {
            pos: [rng.gen_range(-half_w..half_w), rng.gen_range(-half_h..half_h)],
            vel: [0.0, 0.0],
            ptype: rng.gen_range(0..num_types),
            _pad: 0,
        })
        .collect()
}

fn random_matrix(rng: &mut impl Rng, num_types: u32) -> Vec<f32> {
    (0..num_types * num_types)
        .map(|_| rng.gen_range(-1.0f32..1.0))
        .collect()
}

fn model(app: &App) -> Model {
    let w_id = app
        .new_window::<Model>()
        .primary()
        .size(WINDOW_W, WINDOW_H)
        .title("Particle Life")
        .resizable(false)
        .hdr(true)
        .build();

    let window = app.window(w_id);
    let device = window.device();

    let half_w = WINDOW_W as f32 * 0.5;
    let half_h = WINDOW_H as f32 * 0.5;

    let mut rng = rand::thread_rng();
    let particles = random_particles(&mut rng, NUM_PARTICLES, NUM_TYPES, half_w, half_h);
    let matrix = random_matrix(&mut rng, NUM_TYPES);

    let params = Params {
        half_width: half_w,
        half_height: half_h,
        dt: 1.0 / 60.0,
        friction: 0.86,
        max_radius: 80.0,
        beta: 0.3,
        force_scale: 6000.0,
        particle_radius: 2.5,
        num_types: NUM_TYPES,
        num_particles: NUM_PARTICLES,
    };

    let particle_buf_size = std::num::NonZeroU64::new(
        (NUM_PARTICLES as usize * std::mem::size_of::<Particle>()) as u64,
    )
    .unwrap();
    let matrix_buf_size =
        std::num::NonZeroU64::new((matrix.len() * std::mem::size_of::<f32>()) as u64).unwrap();

    let particle_bytes = bytemuck::cast_slice(&particles);
    let particle_buffer_a = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("particles-a"),
        contents: particle_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let particle_buffer_b = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("particles-b"),
        contents: particle_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let matrix_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("interaction-matrix"),
        contents: bytemuck::cast_slice(&matrix),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let params_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // --- compute pipeline ---
    let cs_desc = wgpu::include_wgsl!("shaders/compute.wgsl");
    let cs_mod = device.create_shader_module(cs_desc);

    let compute_bgl = wgpu::BindGroupLayoutBuilder::new()
        .uniform_buffer(wgpu::ShaderStages::COMPUTE, false)
        .storage_buffer(wgpu::ShaderStages::COMPUTE, false, true) // matrix (read-only)
        .storage_buffer(wgpu::ShaderStages::COMPUTE, false, true) // particles_in (read-only)
        .storage_buffer(wgpu::ShaderStages::COMPUTE, false, false) // particles_out (read-write)
        .build(&device);

    let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("particle-life-compute-layout"),
        bind_group_layouts: &[Some(&compute_bgl)],
        immediate_size: 0,
    });
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("particle-life-compute"),
        layout: Some(&compute_pipeline_layout),
        module: &cs_mod,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let make_compute_bind_group = |in_buf: &wgpu::Buffer, out_buf: &wgpu::Buffer| {
        wgpu::BindGroupBuilder::new()
            .buffer::<Params>(&params_buffer, 0..1)
            .buffer_bytes(&matrix_buffer, 0, Some(matrix_buf_size))
            .buffer_bytes(in_buf, 0, Some(particle_buf_size))
            .buffer_bytes(out_buf, 0, Some(particle_buf_size))
            .build(&device, &compute_bgl)
    };
    let compute_bind_group_0 = make_compute_bind_group(&particle_buffer_a, &particle_buffer_b);
    let compute_bind_group_1 = make_compute_bind_group(&particle_buffer_b, &particle_buffer_a);

    // --- render pipeline ---
    let vs_desc = wgpu::include_wgsl!("shaders/vs.wgsl");
    let fs_desc = wgpu::include_wgsl!("shaders/fs.wgsl");
    let vs_mod = device.create_shader_module(vs_desc);
    let fs_mod = device.create_shader_module(fs_desc);

    let render_bgl = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(wgpu::ShaderStages::VERTEX, false, true) // particles (read-only)
        .uniform_buffer(wgpu::ShaderStages::VERTEX, false)
        .build(&device);

    let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("particle-life-render-layout"),
        bind_group_layouts: &[Some(&render_bgl)],
        immediate_size: 0,
    });

    let render_pipeline = wgpu::RenderPipelineBuilder::from_layout(&render_pipeline_layout, &vs_mod)
        .fragment_shader(&fs_mod)
        .color_format(Frame::TEXTURE_FORMAT)
        .add_vertex_buffer::<QuadVertex>(&wgpu::vertex_attr_array![0 => Float32x2])
        .sample_count(window.msaa_samples())
        .build(&device);

    let make_render_bind_group = |buf: &wgpu::Buffer| {
        wgpu::BindGroupBuilder::new()
            .buffer_bytes(buf, 0, Some(particle_buf_size))
            .buffer::<Params>(&params_buffer, 0..1)
            .build(&device, &render_bgl)
    };
    let render_bind_group_0 = make_render_bind_group(&particle_buffer_a);
    let render_bind_group_1 = make_render_bind_group(&particle_buffer_b);

    let quad_vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("quad-vertices"),
        contents: bytemuck::cast_slice(&QUAD_VERTICES),
        usage: wgpu::BufferUsages::VERTEX,
    });

    Model {
        compute_pipeline: Arc::new(compute_pipeline),
        compute_bind_groups: [Arc::new(compute_bind_group_0), Arc::new(compute_bind_group_1)],
        render_pipeline: Arc::new(render_pipeline),
        render_bind_groups: [Arc::new(render_bind_group_0), Arc::new(render_bind_group_1)],
        quad_vertex_buffer: Arc::new(quad_vertex_buffer),
        matrix_buffer: Arc::new(matrix_buffer),
        particle_buffers: [Arc::new(particle_buffer_a), Arc::new(particle_buffer_b)],
        current: 0,
        paused: false,
        params,
    }
}

fn update(app: &App, model: &mut Model) {
    let window = app.main_window();

    if app.keys().just_pressed(KeyCode::Space) {
        model.paused = !model.paused;
    }

    if app.keys().just_pressed(KeyCode::KeyR) {
        let mut rng = rand::thread_rng();
        let particles = random_particles(
            &mut rng,
            model.params.num_particles,
            model.params.num_types,
            model.params.half_width,
            model.params.half_height,
        );
        let matrix = random_matrix(&mut rng, model.params.num_types);
        let particle_bytes = bytemuck::cast_slice(&particles);
        window
            .queue()
            .write_buffer(&model.particle_buffers[0], 0, particle_bytes);
        window
            .queue()
            .write_buffer(&model.particle_buffers[1], 0, particle_bytes);
        window
            .queue()
            .write_buffer(&model.matrix_buffer, 0, bytemuck::cast_slice(&matrix));
        model.current = 0;
    }

    if model.paused {
        return;
    }

    let device = window.device();
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("particle-life-compute"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("particle-life-compute-pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&model.compute_pipeline);
        cpass.set_bind_group(0, &*model.compute_bind_groups[model.current], &[]);
        let workgroups = (model.params.num_particles + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }
    window.queue().submit(Some(encoder.finish()));
    model.current = 1 - model.current;
}

fn render(_app: &RenderApp, model: &Model, frame: Frame) {
    let mut encoder = frame.command_encoder();
    let mut render_pass = wgpu::RenderPassBuilder::new()
        .color_attachment_descriptor(frame.color_attachment(wgpu::LoadOp::Clear(wgpu::Color {
            r: 0.03,
            g: 0.03,
            b: 0.05,
            a: 1.0,
        })))
        .begin(&mut encoder);
    render_pass.set_bind_group(0, &*model.render_bind_groups[model.current], &[]);
    render_pass.set_pipeline(&model.render_pipeline);
    render_pass.set_vertex_buffer(0, model.quad_vertex_buffer.slice(..));

    let vertex_range = 0..QUAD_VERTICES.len() as u32;
    let instance_range = 0..model.params.num_particles;
    render_pass.draw(vertex_range, instance_range);
}
