mod presets;
mod ui;

use bevy_window::{Window as BevyWindow, WindowMode};
use nannou::prelude::*;
use rand::Rng;
use std::sync::Arc;
use ui::{UiLayout, UiVertex};

const WINDOW_W: u32 = 1280;
const WINDOW_H: u32 = 720;
// Simulation world is larger than the screen (same 16:9 ratio, ~10x the
// area) so there's a "canvas" to pan and zoom around in. World size is
// fixed regardless of window size / fullscreen - only the viewport into it
// changes size, so particle count never needs to change on resize.
const WORLD_W: f32 = 4096.0;
const WORLD_H: f32 = 2304.0;
const NUM_PARTICLES: u32 = 130_000;
const NUM_TYPES: u32 = 6;
const WORKGROUP_SIZE: u32 = 64;
// Camera zoom bounds. Min fits the whole world to the initial window size;
// max is an arbitrary "close enough to see individual particles" limit.
const ZOOM_MIN: f32 = 0.28;
const ZOOM_MAX: f32 = 8.0;
// Multiplicative zoom rate per second (exponential: feels equally fast at
// any zoom level) and pan speed in world units/sec at zoom=1.
const ZOOM_RATE: f32 = 1.6;
const PAN_SPEED: f32 = 900.0;
// Spatial grid cell size for neighbor search. Must be >= max_radius so the
// 3x3 neighborhood always covers the full interaction radius.
const CELL_SIZE: f32 = 80.0;
// Hard cap on particles checked per neighbor cell. Particle Life clusters
// particles by design, so a handful of cells can end up wildly overpopulated
// once clumps form; this bounds worst-case per-particle cost regardless.
// Chosen so worst case (9 cells * cap) lands near the pair budget that ran
// comfortably at 40k brute-force particles (~1.6e9 pairs/frame).
const MAX_CELL_SCAN: u32 = 400;
// Default force scale (1/4 of the previous 300.0 default). The speed slider
// multiplies this, and can go as low as 1/32 of it.
const BASE_FORCE_SCALE: f32 = 75.0;
const SPEED_MULT_MIN: f32 = 1.0 / 32.0;
const SPEED_MULT_MAX: f32 = 2.0;

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
    // World half-extents (the simulation canvas: physics, wrap, spatial grid).
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
    cell_size: f32,
    grid_cols: u32,
    grid_rows: u32,
    num_cells: u32,
    max_cell_scan: u32,
    // Screen half-extents (the viewport in pixels) and camera: these turn
    // world-space positions into clip space for the particle vertex shader.
    // The UI panel is screen-locked and never reads these.
    screen_half_w: f32,
    screen_half_h: f32,
    camera_x: f32,
    camera_y: f32,
    zoom: f32,
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
    clear_counts_pipeline: Arc<wgpu::ComputePipeline>,
    count_pipeline: Arc<wgpu::ComputePipeline>,
    prefix_sum_pipeline: Arc<wgpu::ComputePipeline>,
    scatter_pipeline: Arc<wgpu::ComputePipeline>,
    force_pipeline: Arc<wgpu::ComputePipeline>,
    num_cells: u32,
    compute_bind_groups: [Arc<wgpu::BindGroup>; 2],
    render_pipeline: Arc<wgpu::RenderPipeline>,
    render_bind_groups: [Arc<wgpu::BindGroup>; 2],
    quad_vertex_buffer: Arc<wgpu::Buffer>,
    matrix_buffer: Arc<wgpu::Buffer>,
    params_buffer: Arc<wgpu::Buffer>,
    ui_pipeline: Arc<wgpu::RenderPipeline>,
    ui_vertex_buffer: Arc<wgpu::Buffer>,
    ui_vertex_count: u32,
    current: usize,
    paused: bool,
    params: Params,
    matrix: Vec<f32>,
    speed_mult: f32,
    dragging_speed: bool,
    dragging_cell: Option<usize>,
    drag_start_value: f32,
    drag_start_mouse_y: f32,
    preset_index: usize,
    fullscreen: bool,
    pending_resize: bool,
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

// The particle buffers, spatial-grid scratch buffers, and the bind groups
// that reference them - everything needed to run the simulation for a given
// world size / particle count. World size is fixed for the app's lifetime,
// so this only ever runs once, at startup.
struct ParticleResources {
    compute_bind_group_0: wgpu::BindGroup,
    compute_bind_group_1: wgpu::BindGroup,
    render_bind_group_0: wgpu::BindGroup,
    render_bind_group_1: wgpu::BindGroup,
}

fn build_particle_resources(
    device: &wgpu::Device,
    compute_bgl: &wgpu::BindGroupLayout,
    render_bgl: &wgpu::BindGroupLayout,
    params_buffer: &wgpu::Buffer,
    matrix_buffer: &wgpu::Buffer,
    matrix_buf_size: std::num::NonZeroU64,
    half_w: f32,
    half_h: f32,
    num_particles: u32,
    num_types: u32,
    rng: &mut impl Rng,
) -> ParticleResources {
    let grid_cols = (half_w * 2.0 / CELL_SIZE).ceil() as u32;
    let grid_rows = (half_h * 2.0 / CELL_SIZE).ceil() as u32;
    let num_cells = grid_cols * grid_rows;

    let particles = random_particles(rng, num_particles, num_types, half_w, half_h);
    let particle_buf_size =
        std::num::NonZeroU64::new((num_particles as usize * std::mem::size_of::<Particle>()) as u64)
            .unwrap();

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

    let cell_counts_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cell-counts"),
        size: (num_cells as usize * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let cell_offsets_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cell-offsets"),
        size: (num_cells as usize * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let sorted_indices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sorted-indices"),
        size: (num_particles as usize * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let cell_counts_buf_size =
        std::num::NonZeroU64::new((num_cells as usize * std::mem::size_of::<u32>()) as u64).unwrap();
    let sorted_indices_buf_size =
        std::num::NonZeroU64::new((num_particles as usize * std::mem::size_of::<u32>()) as u64)
            .unwrap();

    let make_compute_bind_group = |in_buf: &wgpu::Buffer, out_buf: &wgpu::Buffer| {
        wgpu::BindGroupBuilder::new()
            .buffer::<Params>(params_buffer, 0..1)
            .buffer_bytes(matrix_buffer, 0, Some(matrix_buf_size))
            .buffer_bytes(in_buf, 0, Some(particle_buf_size))
            .buffer_bytes(out_buf, 0, Some(particle_buf_size))
            .buffer_bytes(&cell_counts_buffer, 0, Some(cell_counts_buf_size))
            .buffer_bytes(&cell_offsets_buffer, 0, Some(cell_counts_buf_size))
            .buffer_bytes(&sorted_indices_buffer, 0, Some(sorted_indices_buf_size))
            .build(device, compute_bgl)
    };
    let compute_bind_group_0 = make_compute_bind_group(&particle_buffer_a, &particle_buffer_b);
    let compute_bind_group_1 = make_compute_bind_group(&particle_buffer_b, &particle_buffer_a);

    let make_render_bind_group = |buf: &wgpu::Buffer| {
        wgpu::BindGroupBuilder::new()
            .buffer_bytes(buf, 0, Some(particle_buf_size))
            .buffer::<Params>(params_buffer, 0..1)
            .build(device, render_bgl)
    };
    let render_bind_group_0 = make_render_bind_group(&particle_buffer_a);
    let render_bind_group_1 = make_render_bind_group(&particle_buffer_b);

    ParticleResources {
        compute_bind_group_0,
        compute_bind_group_1,
        render_bind_group_0,
        render_bind_group_1,
    }
}

fn model(app: &App) -> Model {
    let w_id = app
        .new_window::<Model>()
        .primary()
        .size(WINDOW_W, WINDOW_H)
        .title("Particle Life")
        .resizable(true)
        .hdr(true)
        .build();

    let window = app.window(w_id);
    let device = window.device();

    // World: the simulation canvas (fixed size, independent of window size).
    let half_w = WORLD_W * 0.5;
    let half_h = WORLD_H * 0.5;
    // Screen: the viewport in pixels (changes on resize/fullscreen).
    let screen_half_w = WINDOW_W as f32 * 0.5;
    let screen_half_h = WINDOW_H as f32 * 0.5;
    // Start zoomed out just enough to fit the whole world in the window.
    let initial_zoom = (WINDOW_W as f32 / WORLD_W).min(WINDOW_H as f32 / WORLD_H);

    let mut rng = rand::thread_rng();
    let matrix = random_matrix(&mut rng, NUM_TYPES);

    let grid_cols = (WORLD_W / CELL_SIZE).ceil() as u32;
    let grid_rows = (WORLD_H / CELL_SIZE).ceil() as u32;
    let num_cells = grid_cols * grid_rows;

    let params = Params {
        half_width: half_w,
        half_height: half_h,
        dt: 1.0 / 60.0,
        friction: 0.80,
        max_radius: 80.0,
        beta: 0.3,
        force_scale: BASE_FORCE_SCALE,
        particle_radius: 3.0,
        num_types: NUM_TYPES,
        num_particles: NUM_PARTICLES,
        cell_size: CELL_SIZE,
        grid_cols,
        grid_rows,
        num_cells,
        max_cell_scan: MAX_CELL_SCAN,
        screen_half_w,
        screen_half_h,
        camera_x: 0.0,
        camera_y: 0.0,
        zoom: initial_zoom,
    };

    let matrix_buf_size =
        std::num::NonZeroU64::new((matrix.len() * std::mem::size_of::<f32>()) as u64).unwrap();

    let matrix_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("interaction-matrix"),
        contents: bytemuck::cast_slice(&matrix),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let params_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // --- spatial-grid compute pipeline (5 passes: clear, count, prefix sum,
    // scatter, force) sharing one bind group layout and one shader module ---
    let cs_desc = wgpu::include_wgsl!("shaders/grid_compute.wgsl");
    let cs_mod = device.create_shader_module(cs_desc);

    let compute_bgl = wgpu::BindGroupLayoutBuilder::new()
        .uniform_buffer(wgpu::ShaderStages::COMPUTE, false) // 0: params
        .storage_buffer(wgpu::ShaderStages::COMPUTE, false, true) // 1: matrix (read-only)
        .storage_buffer(wgpu::ShaderStages::COMPUTE, false, true) // 2: particles_in (read-only)
        .storage_buffer(wgpu::ShaderStages::COMPUTE, false, false) // 3: particles_out (read-write)
        .storage_buffer(wgpu::ShaderStages::COMPUTE, false, false) // 4: cell_counts (read-write)
        .storage_buffer(wgpu::ShaderStages::COMPUTE, false, false) // 5: cell_offsets (read-write)
        .storage_buffer(wgpu::ShaderStages::COMPUTE, false, false) // 6: sorted_indices (read-write)
        .build(&device);

    let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("particle-life-compute-layout"),
        bind_group_layouts: &[Some(&compute_bgl)],
        immediate_size: 0,
    });

    let make_compute_pipeline = |label: &str, entry_point: &str| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&compute_pipeline_layout),
            module: &cs_mod,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let clear_counts_pipeline = make_compute_pipeline("particle-life-clear-counts", "clear_counts");
    let count_pipeline = make_compute_pipeline("particle-life-count", "count_particles");
    let prefix_sum_pipeline = make_compute_pipeline("particle-life-prefix-sum", "prefix_sum");
    let scatter_pipeline = make_compute_pipeline("particle-life-scatter", "scatter_particles");
    let force_pipeline = make_compute_pipeline("particle-life-force", "compute_forces");

    // --- particle render pipeline ---
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

    let quad_vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("quad-vertices"),
        contents: bytemuck::cast_slice(&QUAD_VERTICES),
        usage: wgpu::BufferUsages::VERTEX,
    });

    // --- on-screen UI overlay pipeline (plain colored triangles, no bind groups) ---
    let ui_vs_desc = wgpu::include_wgsl!("shaders/ui_vs.wgsl");
    let ui_fs_desc = wgpu::include_wgsl!("shaders/ui_fs.wgsl");
    let ui_vs_mod = device.create_shader_module(ui_vs_desc);
    let ui_fs_mod = device.create_shader_module(ui_fs_desc);

    let ui_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("particle-life-ui-layout"),
        bind_group_layouts: &[],
        immediate_size: 0,
    });

    const UI_VERTEX_ATTRS: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x4];
    let ui_pipeline = wgpu::RenderPipelineBuilder::from_layout(&ui_pipeline_layout, &ui_vs_mod)
        .fragment_shader(&ui_fs_mod)
        .color_format(Frame::TEXTURE_FORMAT)
        .add_vertex_buffer::<UiVertex>(&UI_VERTEX_ATTRS)
        .sample_count(window.msaa_samples())
        .build(&device);

    let ui_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ui-vertices"),
        size: (ui::UI_MAX_VERTICES * std::mem::size_of::<UiVertex>()) as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let particle_res = build_particle_resources(
        &device,
        &compute_bgl,
        &render_bgl,
        &params_buffer,
        &matrix_buffer,
        matrix_buf_size,
        half_w,
        half_h,
        NUM_PARTICLES,
        NUM_TYPES,
        &mut rng,
    );

    Model {
        clear_counts_pipeline: Arc::new(clear_counts_pipeline),
        count_pipeline: Arc::new(count_pipeline),
        prefix_sum_pipeline: Arc::new(prefix_sum_pipeline),
        scatter_pipeline: Arc::new(scatter_pipeline),
        force_pipeline: Arc::new(force_pipeline),
        num_cells,
        compute_bind_groups: [
            Arc::new(particle_res.compute_bind_group_0),
            Arc::new(particle_res.compute_bind_group_1),
        ],
        render_pipeline: Arc::new(render_pipeline),
        render_bind_groups: [
            Arc::new(particle_res.render_bind_group_0),
            Arc::new(particle_res.render_bind_group_1),
        ],
        quad_vertex_buffer: Arc::new(quad_vertex_buffer),
        matrix_buffer: Arc::new(matrix_buffer),
        params_buffer: Arc::new(params_buffer),
        ui_pipeline: Arc::new(ui_pipeline),
        ui_vertex_buffer: Arc::new(ui_vertex_buffer),
        ui_vertex_count: 0,
        current: 0,
        paused: false,
        params,
        matrix,
        speed_mult: 1.0,
        dragging_speed: false,
        dragging_cell: None,
        drag_start_value: 0.0,
        drag_start_mouse_y: 0.0,
        preset_index: 0,
        fullscreen: false,
        pending_resize: false,
    }
}

// Toggles fullscreen on F, and once the OS actually resizes the window
// (which takes a frame or more after the mode change), updates the screen
// viewport extents. World size is fixed, so this never touches particle or
// grid buffers - only the screen_half_w/h that the camera transform reads.
// Returns whether Params changed (and needs re-uploading).
fn handle_fullscreen_and_resize(app: &App, model: &mut Model) -> bool {
    if app.keys().just_pressed(KeyCode::KeyF) {
        model.fullscreen = !model.fullscreen;
        let going_fullscreen = model.fullscreen;
        let window_entity = app.window_id();
        app.command_scope(move |mut commands| {
            commands
                .entity(window_entity)
                .entry::<BevyWindow>()
                .and_modify(move |mut w| {
                    w.mode = if going_fullscreen {
                        WindowMode::BorderlessFullscreen(MonitorSelection::Primary)
                    } else {
                        WindowMode::Windowed
                    };
                });
        });
        model.pending_resize = true;
    }

    let mut params_changed = false;
    if model.pending_resize {
        let window = app.main_window();
        let rect = window.rect();
        let new_screen_half_w = rect.w() * 0.5;
        let new_screen_half_h = rect.h() * 0.5;
        if (new_screen_half_w - model.params.screen_half_w).abs() > 0.5
            || (new_screen_half_h - model.params.screen_half_h).abs() > 0.5
        {
            model.params.screen_half_w = new_screen_half_w;
            model.params.screen_half_h = new_screen_half_h;
            params_changed = true;
            model.pending_resize = false;
        }
    }
    params_changed
}

// Keyboard camera controls: zoom (=/-, continuous while held) and pan (arrow
// keys). Pan speed scales with 1/zoom so it feels like a constant on-screen
// speed rather than a constant world-space speed (otherwise panning feels
// sluggish when zoomed in and way too fast when zoomed out). Panning past a
// world edge wraps around, matching how particles themselves wrap.
// Returns whether Params changed (and needs re-uploading).
fn handle_camera_input(app: &App, model: &mut Model) -> bool {
    let mut params_changed = false;
    let dt = app.time_delta();

    if app.keys().pressed(KeyCode::Equal) || app.keys().pressed(KeyCode::NumpadAdd) {
        model.params.zoom = (model.params.zoom * (1.0 + ZOOM_RATE * dt)).clamp(ZOOM_MIN, ZOOM_MAX);
        params_changed = true;
    }
    if app.keys().pressed(KeyCode::Minus) || app.keys().pressed(KeyCode::NumpadSubtract) {
        model.params.zoom = (model.params.zoom / (1.0 + ZOOM_RATE * dt)).clamp(ZOOM_MIN, ZOOM_MAX);
        params_changed = true;
    }

    let pan_step = PAN_SPEED * dt / model.params.zoom;
    let mut pan = Vec2::ZERO;
    if app.keys().pressed(KeyCode::ArrowLeft) {
        pan.x -= pan_step;
    }
    if app.keys().pressed(KeyCode::ArrowRight) {
        pan.x += pan_step;
    }
    if app.keys().pressed(KeyCode::ArrowUp) {
        pan.y += pan_step;
    }
    if app.keys().pressed(KeyCode::ArrowDown) {
        pan.y -= pan_step;
    }
    if pan != Vec2::ZERO {
        let world_w = model.params.half_width * 2.0;
        let world_h = model.params.half_height * 2.0;
        let wrap = |v: f32, half: f32, full: f32| ((v + half).rem_euclid(full)) - half;
        model.params.camera_x = wrap(model.params.camera_x + pan.x, model.params.half_width, world_w);
        model.params.camera_y = wrap(model.params.camera_y + pan.y, model.params.half_height, world_h);
        params_changed = true;
    }

    params_changed
}

// Mouse interaction with the on-screen panel (screen-locked; the panel never
// moves or scales with the world camera): press-to-start-drag, drag-to-set
// for both the speed slider and matrix cells, release-to-stop. Returns
// (params_changed, matrix_changed).
fn handle_ui_mouse(app: &App, model: &mut Model, layout: &UiLayout, num_types: u32) -> (bool, bool) {
    let mut params_changed = false;
    let mut matrix_changed = false;

    let mouse = app.mouse();
    let mouse_pressed = app.mouse_buttons().just_pressed(MouseButton::Left);
    let mouse_down = app.mouse_buttons().pressed(MouseButton::Left);
    let mouse_released = app.mouse_buttons().just_released(MouseButton::Left);

    if mouse_pressed {
        if layout.slider_hit.contains(mouse) {
            model.dragging_speed = true;
        } else {
            for row in 0..num_types {
                for col in 0..num_types {
                    let rect = ui::grid_cell_rect(layout.grid_origin, row + 1, col + 1);
                    if rect.contains(mouse) {
                        let idx = (row * num_types + col) as usize;
                        model.dragging_cell = Some(idx);
                        model.drag_start_value = model.matrix[idx];
                        model.drag_start_mouse_y = mouse.y;
                    }
                }
            }
        }
    }

    if mouse_down && model.dragging_speed {
        let t = ((mouse.x - layout.slider_x0) / (layout.slider_x1 - layout.slider_x0)).clamp(0.0, 1.0);
        let log_min = SPEED_MULT_MIN.ln();
        let log_max = SPEED_MULT_MAX.ln();
        model.speed_mult = (log_min + t * (log_max - log_min)).exp();
        params_changed = true;
    }

    // Value tracks the drag absolutely (start value + offset from the drag's
    // starting point), the same principle as the slider above, rather than
    // accumulating per-frame deltas which feels disconnected from the cursor.
    if mouse_down {
        if let Some(idx) = model.dragging_cell {
            const DRAG_RANGE: f32 = 120.0;
            let new_value =
                (model.drag_start_value + (mouse.y - model.drag_start_mouse_y) / DRAG_RANGE * 2.0)
                    .clamp(-1.0, 1.0);
            if new_value != model.matrix[idx] {
                model.matrix[idx] = new_value;
                matrix_changed = true;
            }
        }
    }

    if mouse_released {
        model.dragging_speed = false;
        model.dragging_cell = None;
    }

    (params_changed, matrix_changed)
}

// Rebuilds the UI overlay geometry and uploads it - runs every frame, even
// while paused, so the panel stays live and interactive at all times.
fn build_and_upload_ui(
    model: &mut Model,
    queue: &wgpu::Queue,
    layout: &UiLayout,
    half_w: f32,
    half_h: f32,
    num_types: u32,
) {
    let mut ui_vertices: Vec<UiVertex> = Vec::with_capacity(ui::UI_MAX_VERTICES);

    ui::push_rect(&mut ui_vertices, &layout.panel_bg, [0.0, 0.0, 0.0, 0.55], half_w, half_h);
    ui::push_rect(&mut ui_vertices, &layout.slider_track, [1.0, 1.0, 1.0, 0.25], half_w, half_h);

    let log_min = SPEED_MULT_MIN.ln();
    let log_max = SPEED_MULT_MAX.ln();
    let t = ((model.speed_mult.ln() - log_min) / (log_max - log_min)).clamp(0.0, 1.0);
    let handle_x = layout.slider_x0 + t * (layout.slider_x1 - layout.slider_x0);
    let handle_half = 6.0;
    let handle_rect = ui::UiRect {
        x0: handle_x - handle_half,
        x1: handle_x + handle_half,
        y0: layout.slider_y_center - handle_half - 3.0,
        y1: layout.slider_y_center + handle_half + 3.0,
    };
    ui::push_rect(&mut ui_vertices, &handle_rect, [1.0, 1.0, 1.0, 0.9], half_w, half_h);

    // Column headers get a downward chevron ("read down this column"); row
    // headers get a rightward chevron ("read across this row") — a text-free
    // way to show which axis is "from" and which is "to" in the matrix.
    const CHEVRON_COLOR: [f32; 4] = [0.05, 0.05, 0.05, 0.85];
    for i in 0..num_types {
        let rect_col = ui::grid_cell_rect(layout.grid_origin, 0, i + 1);
        ui::push_rect(&mut ui_vertices, &rect_col, ui::type_color_rgb(i), half_w, half_h);
        ui::push_down_chevron(&mut ui_vertices, &rect_col, CHEVRON_COLOR, half_w, half_h);

        let rect_row = ui::grid_cell_rect(layout.grid_origin, i + 1, 0);
        ui::push_rect(&mut ui_vertices, &rect_row, ui::type_color_rgb(i), half_w, half_h);
        ui::push_right_chevron(&mut ui_vertices, &rect_row, CHEVRON_COLOR, half_w, half_h);
    }
    for row in 0..num_types {
        for col in 0..num_types {
            let idx = (row * num_types + col) as usize;
            let rect = ui::grid_cell_rect(layout.grid_origin, row + 1, col + 1);
            let mut color = ui::matrix_value_color(model.matrix[idx]);
            if model.dragging_cell == Some(idx) {
                // Brighten the cell currently being dragged for clear feedback.
                for c in color.iter_mut().take(3) {
                    *c = (*c + 0.35).min(1.0);
                }
            }
            ui::push_rect(&mut ui_vertices, &rect, color, half_w, half_h);
        }
    }

    model.ui_vertex_count = ui_vertices.len() as u32;
    queue.write_buffer(&model.ui_vertex_buffer, 0, bytemuck::cast_slice(&ui_vertices));
}

// The 6-pass GPU dispatch: clear -> count -> prefix-sum -> clear -> scatter
// -> force (see grid_compute.wgsl for what each pass does).
fn dispatch_compute(model: &mut Model, device: &wgpu::Device, queue: &wgpu::Queue) {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("particle-life-compute"),
    });
    let bind_group = &*model.compute_bind_groups[model.current];
    let particle_workgroups = (model.params.num_particles + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    let cell_workgroups = (model.num_cells + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

    let dispatch = |encoder: &mut wgpu::CommandEncoder,
                     label: &str,
                     pipeline: &wgpu::ComputePipeline,
                     workgroups: u32| {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, bind_group, &[]);
        cpass.dispatch_workgroups(workgroups, 1, 1);
    };

    // 1. Zero the per-cell histogram.
    dispatch(&mut encoder, "clear-counts", &model.clear_counts_pipeline, cell_workgroups);
    // 2. Count particles per cell.
    dispatch(&mut encoder, "count-particles", &model.count_pipeline, particle_workgroups);
    // 3. Exclusive prefix sum -> per-cell start offsets.
    dispatch(&mut encoder, "prefix-sum", &model.prefix_sum_pipeline, 1);
    // 4. Zero counts again, to reuse as a per-cell write cursor.
    dispatch(&mut encoder, "clear-counts-2", &model.clear_counts_pipeline, cell_workgroups);
    // 5. Scatter particle indices into cell-sorted order.
    dispatch(&mut encoder, "scatter-particles", &model.scatter_pipeline, particle_workgroups);
    // 6. Force pass: each particle only checks its 3x3 neighboring cells.
    dispatch(&mut encoder, "compute-forces", &model.force_pipeline, particle_workgroups);

    queue.submit(Some(encoder.finish()));
    model.current = 1 - model.current;
}

fn update(app: &App, model: &mut Model) {
    if app.keys().just_pressed(KeyCode::KeyP) {
        model.paused = !model.paused;
    }

    let mut params_changed = handle_fullscreen_and_resize(app, model);
    params_changed |= handle_camera_input(app, model);

    let randomize_matrix_key = app.keys().just_pressed(KeyCode::KeyR);
    let cycle_preset = app.keys().just_pressed(KeyCode::Space);

    let half_w = model.params.screen_half_w;
    let half_h = model.params.screen_half_h;
    let num_types = model.params.num_types;
    let layout = ui::compute_ui_layout(half_w, half_h, num_types);

    let (ui_params_changed, mut matrix_changed) = handle_ui_mouse(app, model, &layout, num_types);
    params_changed |= ui_params_changed;

    let window = app.main_window();
    let device = window.device();
    let queue = window.queue();

    if params_changed {
        model.params.force_scale = BASE_FORCE_SCALE * model.speed_mult;
        queue.write_buffer(&model.params_buffer, 0, bytemuck::bytes_of(&model.params));
    }

    let mut rng = rand::thread_rng();
    if randomize_matrix_key {
        model.matrix = random_matrix(&mut rng, model.params.num_types);
        matrix_changed = true;
    }
    if cycle_preset {
        model.preset_index = (model.preset_index + 1) % presets::PRESET_COUNT;
        model.matrix = presets::preset_matrix(model.preset_index, model.params.num_types);
        matrix_changed = true;
    }
    if matrix_changed {
        queue.write_buffer(&model.matrix_buffer, 0, bytemuck::cast_slice(&model.matrix));
    }

    build_and_upload_ui(model, queue, &layout, half_w, half_h, num_types);

    if model.paused {
        return;
    }

    dispatch_compute(model, device, queue);
}

fn render(_app: &RenderApp, model: &Model, frame: Frame) {
    let mut encoder = frame.command_encoder();
    let mut render_pass = wgpu::RenderPassBuilder::new()
        .color_attachment_descriptor(frame.color_attachment(wgpu::LoadOp::Clear(wgpu::Color {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 1.0,
        })))
        .begin(&mut encoder);

    render_pass.set_bind_group(0, &*model.render_bind_groups[model.current], &[]);
    render_pass.set_pipeline(&model.render_pipeline);
    render_pass.set_vertex_buffer(0, model.quad_vertex_buffer.slice(..));
    let vertex_range = 0..QUAD_VERTICES.len() as u32;
    let instance_range = 0..model.params.num_particles;
    render_pass.draw(vertex_range, instance_range);

    if model.ui_vertex_count > 0 {
        render_pass.set_pipeline(&model.ui_pipeline);
        render_pass.set_vertex_buffer(0, model.ui_vertex_buffer.slice(..));
        render_pass.draw(0..model.ui_vertex_count, 0..1);
    }
}
