struct Particle {
    pos: vec2<f32>,
    vel: vec2<f32>,
    ptype: u32,
    _pad: u32,
};

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
};

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: Params;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec3<f32>,
};

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let c = v * s;
    let hp = h * 6.0;
    let x = c * (1.0 - abs(hp % 2.0 - 1.0));
    var rgb: vec3<f32>;
    if (hp < 1.0) { rgb = vec3<f32>(c, x, 0.0); }
    else if (hp < 2.0) { rgb = vec3<f32>(x, c, 0.0); }
    else if (hp < 3.0) { rgb = vec3<f32>(0.0, c, x); }
    else if (hp < 4.0) { rgb = vec3<f32>(0.0, x, c); }
    else if (hp < 5.0) { rgb = vec3<f32>(x, 0.0, c); }
    else { rgb = vec3<f32>(c, 0.0, x); }
    let m = v - c;
    return rgb + vec3<f32>(m, m, m);
}

// Golden-angle hue spread (via the golden ratio conjugate) so adjacent type
// indices land on maximally distinct hues; full saturation/value for vivid
// colors that read clearly against a black background.
fn type_color(t: u32) -> vec3<f32> {
    let hue = fract(f32(t) * 0.6180339887);
    return hsv_to_rgb(hue, 0.85, 1.0);
}

@vertex
fn main(
    @location(0) corner: vec2<f32>,
    @builtin(instance_index) instance: u32,
) -> VertexOutput {
    let p = particles[instance];
    let world_pos = p.pos + corner * params.particle_radius;

    var out: VertexOutput;
    out.clip_position = vec4<f32>(
        world_pos.x / params.half_width,
        world_pos.y / params.half_height,
        0.0,
        1.0,
    );
    out.uv = corner;
    out.color = type_color(p.ptype);
    return out;
}
