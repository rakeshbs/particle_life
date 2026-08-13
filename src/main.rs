use nannou::prelude::*;
use rand::Rng;
use std::sync::Arc;

const WINDOW_W: u32 = 1280;
const WINDOW_H: u32 = 720;
const NUM_PARTICLES: u32 = 20_000;
const NUM_TYPES: u32 = 6;
const WORKGROUP_SIZE: u32 = 64;
// Spatial grid cell size for neighbor search. Must be >= max_radius so the
// 3x3 neighborhood always covers the full interaction radius.
const CELL_SIZE: f32 = 80.0;
// Hard cap on particles checked per neighbor cell. Particle Life clusters
// particles by design, so a handful of cells can end up wildly overpopulated
// once clumps form; this bounds worst-case per-particle cost regardless.
// Chosen so worst case (9 cells * cap) lands near the pair budget that ran
// comfortably at 40k brute-force particles (~1.6e9 pairs/frame).
const MAX_CELL_SCAN: u32 = 400;
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
    cell_size: f32,
    grid_cols: u32,
    grid_rows: u32,
    num_cells: u32,
    max_cell_scan: u32,
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

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let c = v * s;
    let hp = h * 6.0;
    let x = c * (1.0 - (hp % 2.0 - 1.0).abs());
    let rgb = if hp < 1.0 {
        [c, x, 0.0]
    } else if hp < 2.0 {
        [x, c, 0.0]
    } else if hp < 3.0 {
        [0.0, c, x]
    } else if hp < 4.0 {
        [0.0, x, c]
    } else if hp < 5.0 {
        [x, 0.0, c]
    } else {
        [c, 0.0, x]
    };
    let m = v - c;
    [rgb[0] + m, rgb[1] + m, rgb[2] + m]
}

// Golden-angle hue spread, matching type_color() in vs.wgsl, so the grid's
// header swatches match the particle colors on screen. Desaturated toward
// gray so the headers read as a legend rather than competing for attention.
fn type_color_rgb(t: u32) -> [f32; 4] {
    let hue = (t as f32 * 0.6180339887).fract();
    let [r, g, b] = hsv_to_rgb(hue, 0.85, 1.0);
    let sat = 0.55;
    let gray = 0.55;
    [
        gray + (r - gray) * sat,
        gray + (g - gray) * sat,
        gray + (b - gray) * sat,
        1.0,
    ]
}

// Muted diverging scale for matrix cells: dark neutral gray at 0, soft rust
// toward repel, soft teal toward attract.
fn matrix_value_color(v: f32) -> [f32; 4] {
    const NEUTRAL: [f32; 3] = [0.24, 0.24, 0.27];
    const POS: [f32; 3] = [0.25, 0.55, 0.48];
    const NEG: [f32; 3] = [0.55, 0.30, 0.28];
    let t = v.clamp(-1.0, 1.0);
    let target = if t >= 0.0 { POS } else { NEG };
    let a = t.abs();
    [
        NEUTRAL[0] + (target[0] - NEUTRAL[0]) * a,
        NEUTRAL[1] + (target[1] - NEUTRAL[1]) * a,
        NEUTRAL[2] + (target[2] - NEUTRAL[2]) * a,
        1.0,
    ]
}

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

// All presets below are built around asymmetric attraction — a type chases
// another type which only weakly (or doesn't) chase back. That asymmetry is
// what makes a bonded group self-propel and drift across the screen instead
// of settling into a static cluster, so every preset here produces gliders
// of one flavor or another.
const PRESET_COUNT: usize = 8;

fn pair_partner(a: u32) -> u32 {
    if a % 2 == 0 { a + 1 } else { a - 1 }
}

// Three independent pairs (0-1, 2-3, 4-5), strong chase / strong flee: small,
// fast-moving gliders.
fn preset_fast_pairs(n: u32) -> Vec<f32> {
    (0..n * n)
        .map(|idx| {
            let a = idx / n;
            let b = idx % n;
            let partner = pair_partner(a);
            if a == b || partner >= n || b != partner {
                0.0
            } else if a % 2 == 0 {
                0.95
            } else {
                -0.5
            }
        })
        .collect()
}

// Same pairing as Fast Pairs, but gentler asymmetry: slower, more graceful
// drifting gliders.
fn preset_slow_pairs(n: u32) -> Vec<f32> {
    (0..n * n)
        .map(|idx| {
            let a = idx / n;
            let b = idx % n;
            let partner = pair_partner(a);
            if a == b || partner >= n || b != partner {
                0.0
            } else if a % 2 == 0 {
                0.5
            } else {
                -0.15
            }
        })
        .collect()
}

// Pairs with added self-attraction, so each glider stays a tight, solid blob
// while it drifts rather than a loose diffuse pair.
fn preset_cohesive_gliders(n: u32) -> Vec<f32> {
    (0..n * n)
        .map(|idx| {
            let a = idx / n;
            let b = idx % n;
            if a == b {
                return 0.35;
            }
            let partner = pair_partner(a);
            if partner >= n || b != partner {
                return 0.0;
            }
            if a % 2 == 0 { 0.85 } else { -0.4 }
        })
        .collect()
}

// Pairs with strong self-cohesion AND strong asymmetry: compact, dense,
// fast-darting gliders rather than loose clouds.
fn preset_tight_darts(n: u32) -> Vec<f32> {
    (0..n * n)
        .map(|idx| {
            let a = idx / n;
            let b = idx % n;
            if a == b {
                return 0.6;
            }
            let partner = pair_partner(a);
            if partner >= n || b != partner {
                return 0.0;
            }
            if a % 2 == 0 { 0.9 } else { -0.6 }
        })
        .collect()
}

// Independent groups of 3 (0-1-2, 3-4-5, ...), each cycling A chases B chases
// C chases A: the asymmetric triangle doesn't cancel out, so each trio spins
// and drifts together as a small orbiting glider cluster.
fn preset_triad_chasers(n: u32) -> Vec<f32> {
    (0..n * n)
        .map(|idx| {
            let a = idx / n;
            let b = idx % n;
            if a == b {
                return 0.0;
            }
            let group = (a / 3) * 3;
            let pos = a % 3;
            let next = group + (pos + 1) % 3;
            let prev = group + (pos + 2) % 3;
            if b == next {
                0.85
            } else if b == prev {
                -0.3
            } else {
                0.0
            }
        })
        .collect()
}

// One open (non-wrapping) chain across all types. Strong self-cohesion (0.6)
// means each type clumps into a thick, solid segment instead of thin
// scattered points; segments pull the next one forward and push off the
// previous one to walk in a line; and a mild repulsion between every other
// pairing keeps non-adjacent segments from merging into a formless blob, so
// the whole thing reads as one large, distinctly segmented worm.
fn preset_worm_train(n: u32) -> Vec<f32> {
    (0..n * n)
        .map(|idx| {
            let a = (idx / n) as i32;
            let b = (idx % n) as i32;
            if a == b {
                0.6
            } else if b == a + 1 {
                0.75
            } else if a == b + 1 {
                -0.35
            } else {
                -0.15
            }
        })
        .collect()
}

// Same thick-segment recipe as Worm Train, but split into two independent
// open 3-chains (0-1-2 and 3-4-5): two large worms instead of one.
fn preset_twin_worms(n: u32) -> Vec<f32> {
    (0..n * n)
        .map(|idx| {
            let a = idx / n;
            let b = idx % n;
            if a == b {
                return 0.6;
            }
            let group = a / 3;
            let pos = a % 3;
            let same_group = b / 3 == group;
            if same_group && pos < 2 && b == a + 1 {
                0.75
            } else if same_group && pos > 0 && a == b + 1 {
                -0.35
            } else {
                -0.15
            }
        })
        .collect()
}

// Three pairs at three different speeds (fast, medium, slow), plus a weak
// universal attraction so the three gliders loosely stay near each other: a
// small swarm of differently-paced movers instead of one uniform speed.
fn preset_swarm_chase(n: u32) -> Vec<f32> {
    (0..n * n)
        .map(|idx| {
            let a = idx / n;
            let b = idx % n;
            if a == b {
                return 0.0;
            }
            let partner = pair_partner(a);
            if partner < n && b == partner {
                let speed = a / 2; // 0, 1, 2 for the three pairs
                let chase = [0.95, 0.65, 0.35][speed as usize % 3];
                let flee = [-0.6, -0.3, -0.1][speed as usize % 3];
                if a % 2 == 0 { chase } else { flee }
            } else {
                0.08
            }
        })
        .collect()
}

fn preset_matrix(index: usize, num_types: u32) -> Vec<f32> {
    match index % PRESET_COUNT {
        0 => preset_fast_pairs(num_types),
        1 => preset_slow_pairs(num_types),
        2 => preset_cohesive_gliders(num_types),
        3 => preset_tight_darts(num_types),
        4 => preset_triad_chasers(num_types),
        5 => preset_worm_train(num_types),
        6 => preset_twin_worms(num_types),
        _ => preset_swarm_chase(num_types),
    }
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

    let grid_cols = (WINDOW_W as f32 / CELL_SIZE).ceil() as u32;
    let grid_rows = (WINDOW_H as f32 / CELL_SIZE).ceil() as u32;
    let num_cells = grid_cols * grid_rows;

    let params = Params {
        half_width: half_w,
        half_height: half_h,
        dt: 1.0 / 60.0,
        friction: 0.80,
        max_radius: 80.0,
        beta: 0.3,
        force_scale: BASE_FORCE_SCALE,
        particle_radius: 2.0,
        num_types: NUM_TYPES,
        num_particles: NUM_PARTICLES,
        cell_size: CELL_SIZE,
        grid_cols,
        grid_rows,
        num_cells,
        max_cell_scan: MAX_CELL_SCAN,
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

    // --- spatial-grid compute pipeline (5 passes: clear, count, prefix sum,
    // scatter, force) sharing one bind group layout and one shader module ---
    let cs_desc = wgpu::include_wgsl!("shaders/grid_compute.wgsl");
    let cs_mod = device.create_shader_module(cs_desc);

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
        size: (NUM_PARTICLES as usize * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let cell_counts_buf_size =
        std::num::NonZeroU64::new((num_cells as usize * std::mem::size_of::<u32>()) as u64).unwrap();
    let sorted_indices_buf_size =
        std::num::NonZeroU64::new((NUM_PARTICLES as usize * std::mem::size_of::<u32>()) as u64)
            .unwrap();

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

    let make_compute_bind_group = |in_buf: &wgpu::Buffer, out_buf: &wgpu::Buffer| {
        wgpu::BindGroupBuilder::new()
            .buffer::<Params>(&params_buffer, 0..1)
            .buffer_bytes(&matrix_buffer, 0, Some(matrix_buf_size))
            .buffer_bytes(in_buf, 0, Some(particle_buf_size))
            .buffer_bytes(out_buf, 0, Some(particle_buf_size))
            .buffer_bytes(&cell_counts_buffer, 0, Some(cell_counts_buf_size))
            .buffer_bytes(&cell_offsets_buffer, 0, Some(cell_counts_buf_size))
            .buffer_bytes(&sorted_indices_buffer, 0, Some(sorted_indices_buf_size))
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
        clear_counts_pipeline: Arc::new(clear_counts_pipeline),
        count_pipeline: Arc::new(count_pipeline),
        prefix_sum_pipeline: Arc::new(prefix_sum_pipeline),
        scatter_pipeline: Arc::new(scatter_pipeline),
        force_pipeline: Arc::new(force_pipeline),
        num_cells,
        compute_bind_groups: [Arc::new(compute_bind_group_0), Arc::new(compute_bind_group_1)],
        render_pipeline: Arc::new(render_pipeline),
        render_bind_groups: [Arc::new(render_bind_group_0), Arc::new(render_bind_group_1)],
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
    }
}

fn update(app: &App, model: &mut Model) {
    let window = app.main_window();

    if app.keys().just_pressed(KeyCode::KeyP) {
        model.paused = !model.paused;
    }

    let randomize_matrix_key = app.keys().just_pressed(KeyCode::KeyR);
    let cycle_preset = app.keys().just_pressed(KeyCode::Space);
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
    if cycle_preset {
        model.preset_index = (model.preset_index + 1) % PRESET_COUNT;
        model.matrix = preset_matrix(model.preset_index, model.params.num_types);
        matrix_changed = true;
    }
    if matrix_changed {
        window
            .queue()
            .write_buffer(&model.matrix_buffer, 0, bytemuck::cast_slice(&model.matrix));
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

    window.queue().submit(Some(encoder.finish()));
    model.current = 1 - model.current;
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
