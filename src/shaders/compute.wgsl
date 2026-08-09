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

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> matrix: array<f32>;
@group(0) @binding(2) var<storage, read> particles_in: array<Particle>;
@group(0) @binding(3) var<storage, read_write> particles_out: array<Particle>;

// Ventrella-style particle-life force curve: short-range repulsion below
// `beta`, ramping into the (possibly attractive) type-pair force out to r=1.
fn attraction(r: f32, a: f32, beta: f32) -> f32 {
    if (r < beta) {
        return r / beta - 1.0;
    } else if (r < 1.0) {
        return a * (1.0 - abs(2.0 * r - 1.0 - beta) / (1.0 - beta));
    }
    return 0.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) {
        return;
    }

    let p = particles_in[i];
    var accel = vec2<f32>(0.0, 0.0);
    let world_w = params.half_width * 2.0;
    let world_h = params.half_height * 2.0;

    for (var j: u32 = 0u; j < params.num_particles; j = j + 1u) {
        if (j == i) {
            continue;
        }
        let o = particles_in[j];
        var d = o.pos - p.pos;
        // shortest vector on the toroidal world
        if (d.x > params.half_width) { d.x = d.x - world_w; }
        else if (d.x < -params.half_width) { d.x = d.x + world_w; }
        if (d.y > params.half_height) { d.y = d.y - world_h; }
        else if (d.y < -params.half_height) { d.y = d.y + world_h; }

        let dist = length(d);
        if (dist > 0.0001 && dist < params.max_radius) {
            let a = matrix[p.ptype * params.num_types + o.ptype];
            let f = attraction(dist / params.max_radius, a, params.beta);
            accel = accel + (d / dist) * f;
        }
    }

    accel = accel * params.force_scale;
    let vel = (p.vel + accel * params.dt) * params.friction;
    var pos = p.pos + vel * params.dt;

    if (pos.x > params.half_width) { pos.x = pos.x - world_w; }
    if (pos.x < -params.half_width) { pos.x = pos.x + world_w; }
    if (pos.y > params.half_height) { pos.y = pos.y - world_h; }
    if (pos.y < -params.half_height) { pos.y = pos.y + world_h; }

    var out: Particle;
    out.pos = pos;
    out.vel = vel;
    out.ptype = p.ptype;
    out._pad = 0u;
    particles_out[i] = out;
}
