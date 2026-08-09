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
};

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: Params;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec3<f32>,
};

// Golden-angle hue spread so adjacent type indices get visually distinct colors.
fn type_color(t: u32) -> vec3<f32> {
    let hue = f32(t) * 2.399963;
    return vec3<f32>(
        0.5 + 0.5 * cos(hue),
        0.5 + 0.5 * cos(hue + 2.094395),
        0.5 + 0.5 * cos(hue + 4.18879),
    );
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
