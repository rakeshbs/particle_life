use nannou::prelude::*;
use rand::Rng;
use std::sync::Arc;

const WINDOW_W: u32 = 1280;
const WINDOW_H: u32 = 720;
const NUM_PARTICLES: u32 = 4000;
const NUM_TYPES: u32 = 6;
const WORKGROUP_SIZE: u32 = 64;
// Speed slider multiplies this base; can go as low as 1/32 of it.
const BASE_FORCE_SCALE: f32 = 75.0;
const SPEED_MULT_MIN: f32 = 1.0 / 32.0;
const SPEED_MULT_MAX: f32 = 2.0;

// On-screen control panel layout, in the same pixel/world units as particle
// positions (window is not resizable, so fixed pixel geometry is fine).
const UI_MARGIN: f32 = 16.0;
const UI_CELL: f32 = 22.0;
const UI_GAP: f32 = 3.0;
const UI_SLIDER_W: f32 = 200.0;
const UI_SLIDER_H: f32 = 10.0;
const UI_ROW_GAP: f32 = 18.0;
const UI_MAX_VERTICES: usize = 1024;

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

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct UiVertex {
    clip_pos: [f32; 2],
    color: [f32; 4],
}

#[derive(Clone, Copy)]
struct UiRect {
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
}

impl UiRect {
    fn contains(&self, p: Vec2) -> bool {
        p.x >= self.x0 && p.x <= self.x1 && p.y >= self.y0 && p.y <= self.y1
    }
}

struct UiLayout {
    slider_hit: UiRect,
    slider_track: UiRect,
    slider_x0: f32,
    slider_x1: f32,
    slider_y_center: f32,
    grid_origin: (f32, f32),
    panel_bg: UiRect,
}

fn compute_ui_layout(half_w: f32, half_h: f32, num_types: u32) -> UiLayout {
    let panel_left = -half_w + UI_MARGIN;
    let panel_top = half_h - UI_MARGIN;

    let slider_x0 = panel_left;
    let slider_x1 = panel_left + UI_SLIDER_W;
    let slider_y_center = panel_top - UI_SLIDER_H * 0.5;
    let slider_track = UiRect {
        x0: slider_x0,
        x1: slider_x1,
        y0: slider_y_center - UI_SLIDER_H * 0.5,
        y1: slider_y_center + UI_SLIDER_H * 0.5,
    };
    let slider_hit = UiRect {
        x0: slider_x0 - 4.0,
        x1: slider_x1 + 4.0,
        y0: slider_y_center - 10.0,
        y1: slider_y_center + 10.0,
    };

    let grid_top = slider_y_center - UI_SLIDER_H * 0.5 - UI_ROW_GAP;
    let grid_origin = (panel_left, grid_top);

    let grid_cols = num_types + 1;
    let grid_rows = num_types + 1;
    let grid_w = grid_cols as f32 * UI_CELL + (grid_cols - 1) as f32 * UI_GAP;
    let grid_h = grid_rows as f32 * UI_CELL + (grid_rows - 1) as f32 * UI_GAP;
    let panel_bg = UiRect {
        x0: panel_left - 10.0,
        x1: (panel_left + grid_w).max(slider_x1) + 10.0,
        y0: grid_top - grid_h - 10.0,
        y1: panel_top + 10.0,
    };

    UiLayout {
        slider_hit,
        slider_track,
        slider_x0,
        slider_x1,
        slider_y_center,
        grid_origin,
        panel_bg,
    }
}

// row 0 / col 0 are the type-color header swatches; interior cells (row,col
// both >= 1) are the actual interaction-matrix entries.
fn grid_cell_rect(origin: (f32, f32), row: u32, col: u32) -> UiRect {
    let x0 = origin.0 + col as f32 * (UI_CELL + UI_GAP);
    let y1 = origin.1 - row as f32 * (UI_CELL + UI_GAP);
    UiRect {
        x0,
        x1: x0 + UI_CELL,
        y0: y1 - UI_CELL,
        y1,
    }
}

fn push_triangle(
    verts: &mut Vec<UiVertex>,
    p0: (f32, f32),
    p1: (f32, f32),
    p2: (f32, f32),
    color: [f32; 4],
    half_w: f32,
    half_h: f32,
) {
    let c = |x: f32, y: f32| [x / half_w, y / half_h];
    verts.push(UiVertex { clip_pos: c(p0.0, p0.1), color });
    verts.push(UiVertex { clip_pos: c(p1.0, p1.1), color });
    verts.push(UiVertex { clip_pos: c(p2.0, p2.1), color });
}

// Downward chevron overlay for column headers ("read down this column").
fn push_down_chevron(verts: &mut Vec<UiVertex>, rect: &UiRect, color: [f32; 4], half_w: f32, half_h: f32) {
    let cx = (rect.x0 + rect.x1) * 0.5;
    push_triangle(
        verts,
        (cx, rect.y0 + 3.0),
        (cx - 5.0, rect.y1 - 3.0),
        (cx + 5.0, rect.y1 - 3.0),
        color,
        half_w,
        half_h,
    );
}

// Rightward chevron overlay for row headers ("read across this row").
fn push_right_chevron(verts: &mut Vec<UiVertex>, rect: &UiRect, color: [f32; 4], half_w: f32, half_h: f32) {
    let cy = (rect.y0 + rect.y1) * 0.5;
    push_triangle(
        verts,
        (rect.x1 - 3.0, cy),
        (rect.x0 + 3.0, cy + 5.0),
        (rect.x0 + 3.0, cy - 5.0),
        color,
        half_w,
        half_h,
    );
}

fn push_rect(verts: &mut Vec<UiVertex>, rect: &UiRect, color: [f32; 4], half_w: f32, half_h: f32) {
    let c = |x: f32, y: f32| [x / half_w, y / half_h];
    let p00 = c(rect.x0, rect.y0);
    let p10 = c(rect.x1, rect.y0);
    let p11 = c(rect.x1, rect.y1);
    let p01 = c(rect.x0, rect.y1);
    verts.push(UiVertex { clip_pos: p00, color });
    verts.push(UiVertex { clip_pos: p10, color });
    verts.push(UiVertex { clip_pos: p11, color });
    verts.push(UiVertex { clip_pos: p00, color });
    verts.push(UiVertex { clip_pos: p11, color });
    verts.push(UiVertex { clip_pos: p01, color });
}

// Golden-angle hue spread, matching type_color() in vs.wgsl, so the grid's
// header swatches match the particle colors on screen. Desaturated toward
// gray so the headers read as a legend rather than competing for attention.
fn type_color_rgb(t: u32) -> [f32; 4] {
    let hue = t as f32 * 2.399963;
    let r = 0.5 + 0.5 * hue.cos();
    let g = 0.5 + 0.5 * (hue + 2.094395).cos();
    let b = 0.5 + 0.5 * (hue + 4.18879).cos();
    let sat = 0.55;
    let gray = 0.55;
    [
        gray + (r - gray) * sat,
        gray + (g - gray) * sat,
        gray + (b - gray) * sat,
        1.0,
    ]
}

// Diverging scale for matrix cells: red (repel) -> dark (neutral) -> green (attract).
fn matrix_value_color(v: f32) -> [f32; 4] {
    if v >= 0.0 {
        let t = v.clamp(0.0, 1.0);
        [20.0 / 255.0, (20.0 + t * 200.0) / 255.0, 60.0 / 255.0, 1.0]
    } else {
        let t = (-v).clamp(0.0, 1.0);
        [(20.0 + t * 200.0) / 255.0, 20.0 / 255.0, 40.0 / 255.0, 1.0]
    }
}

#[derive(Clone)]
struct Model {
    compute_pipeline: Arc<wgpu::ComputePipeline>,
    compute_bind_groups: [Arc<wgpu::BindGroup>; 2],
    render_pipeline: Arc<wgpu::RenderPipeline>,
    render_bind_groups: [Arc<wgpu::BindGroup>; 2],
    quad_vertex_buffer: Arc<wgpu::Buffer>,
    matrix_buffer: Arc<wgpu::Buffer>,
    params_buffer: Arc<wgpu::Buffer>,
    particle_buffers: [Arc<wgpu::Buffer>; 2],
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
        friction: 0.80,
        max_radius: 80.0,
        beta: 0.3,
        force_scale: BASE_FORCE_SCALE,
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
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
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
        size: (UI_MAX_VERTICES * std::mem::size_of::<UiVertex>()) as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    Model {
        compute_pipeline: Arc::new(compute_pipeline),
        compute_bind_groups: [Arc::new(compute_bind_group_0), Arc::new(compute_bind_group_1)],
        render_pipeline: Arc::new(render_pipeline),
        render_bind_groups: [Arc::new(render_bind_group_0), Arc::new(render_bind_group_1)],
        quad_vertex_buffer: Arc::new(quad_vertex_buffer),
        matrix_buffer: Arc::new(matrix_buffer),
        params_buffer: Arc::new(params_buffer),
        particle_buffers: [Arc::new(particle_buffer_a), Arc::new(particle_buffer_b)],
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
    }
}

fn update(app: &App, model: &mut Model) {
    let window = app.main_window();

    if app.keys().just_pressed(KeyCode::Space) {
        model.paused = !model.paused;
    }

    let reseed_positions = app.keys().just_pressed(KeyCode::KeyR);
    let randomize_matrix_key = app.keys().just_pressed(KeyCode::KeyR);
    let mut matrix_changed = false;
    let mut speed_changed = false;

    // --- mouse interaction with the on-screen panel ---
    let half_w = model.params.half_width;
    let half_h = model.params.half_height;
    let num_types = model.params.num_types;
    let layout = compute_ui_layout(half_w, half_h, num_types);

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
                    let rect = grid_cell_rect(layout.grid_origin, row + 1, col + 1);
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
        speed_changed = true;
    }

    // Value tracks the drag absolutely (start value + offset from the drag's
    // starting point), matching the slider above, rather than accumulating
    // per-frame deltas which feels disconnected from the cursor.
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

    if speed_changed {
        model.params.force_scale = BASE_FORCE_SCALE * model.speed_mult;
        window
            .queue()
            .write_buffer(&model.params_buffer, 0, bytemuck::bytes_of(&model.params));
    }

    let mut rng = rand::thread_rng();
    if randomize_matrix_key {
        model.matrix = random_matrix(&mut rng, model.params.num_types);
        matrix_changed = true;
    }
    if matrix_changed {
        window
            .queue()
            .write_buffer(&model.matrix_buffer, 0, bytemuck::cast_slice(&model.matrix));
    }
    if reseed_positions {
        let particles = random_particles(
            &mut rng,
            model.params.num_particles,
            model.params.num_types,
            model.params.half_width,
            model.params.half_height,
        );
        let particle_bytes = bytemuck::cast_slice(&particles);
        window
            .queue()
            .write_buffer(&model.particle_buffers[0], 0, particle_bytes);
        window
            .queue()
            .write_buffer(&model.particle_buffers[1], 0, particle_bytes);
        model.current = 0;
    }

    // --- rebuild the UI overlay geometry every frame (even while paused) ---
    let mut ui_vertices: Vec<UiVertex> = Vec::with_capacity(UI_MAX_VERTICES);

    push_rect(&mut ui_vertices, &layout.panel_bg, [0.0, 0.0, 0.0, 0.55], half_w, half_h);
    push_rect(&mut ui_vertices, &layout.slider_track, [1.0, 1.0, 1.0, 0.25], half_w, half_h);

    let log_min = SPEED_MULT_MIN.ln();
    let log_max = SPEED_MULT_MAX.ln();
    let t = ((model.speed_mult.ln() - log_min) / (log_max - log_min)).clamp(0.0, 1.0);
    let handle_x = layout.slider_x0 + t * (layout.slider_x1 - layout.slider_x0);
    let handle_half = 6.0;
    let handle_rect = UiRect {
        x0: handle_x - handle_half,
        x1: handle_x + handle_half,
        y0: layout.slider_y_center - handle_half - 3.0,
        y1: layout.slider_y_center + handle_half + 3.0,
    };
    push_rect(&mut ui_vertices, &handle_rect, [1.0, 1.0, 1.0, 0.9], half_w, half_h);

    // Column headers get a downward chevron ("read down this column"); row
    // headers get a rightward chevron ("read across this row") — a text-free
    // way to show which axis is "from" and which is "to" in the matrix.
    const CHEVRON_COLOR: [f32; 4] = [0.05, 0.05, 0.05, 0.85];
    for i in 0..num_types {
        let rect_col = grid_cell_rect(layout.grid_origin, 0, i + 1);
        push_rect(&mut ui_vertices, &rect_col, type_color_rgb(i), half_w, half_h);
        push_down_chevron(&mut ui_vertices, &rect_col, CHEVRON_COLOR, half_w, half_h);

        let rect_row = grid_cell_rect(layout.grid_origin, i + 1, 0);
        push_rect(&mut ui_vertices, &rect_row, type_color_rgb(i), half_w, half_h);
        push_right_chevron(&mut ui_vertices, &rect_row, CHEVRON_COLOR, half_w, half_h);
    }
    for row in 0..num_types {
        for col in 0..num_types {
            let idx = (row * num_types + col) as usize;
            let rect = grid_cell_rect(layout.grid_origin, row + 1, col + 1);
            let mut color = matrix_value_color(model.matrix[idx]);
            if model.dragging_cell == Some(idx) {
                for c in color.iter_mut().take(3) {
                    *c = (*c + 0.35).min(1.0);
                }
            }
            push_rect(&mut ui_vertices, &rect, color, half_w, half_h);
        }
    }

    model.ui_vertex_count = ui_vertices.len() as u32;
    window
        .queue()
        .write_buffer(&model.ui_vertex_buffer, 0, bytemuck::cast_slice(&ui_vertices));

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

    if model.ui_vertex_count > 0 {
        render_pass.set_pipeline(&model.ui_pipeline);
        render_pass.set_vertex_buffer(0, model.ui_vertex_buffer.slice(..));
        render_pass.draw(0..model.ui_vertex_count, 0..1);
    }
}
