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
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> matrix: array<f32>;
@group(0) @binding(2) var<storage, read> particles_in: array<Particle>;
@group(0) @binding(3) var<storage, read_write> particles_out: array<Particle>;
// Per-cell particle count. Used both as a histogram (during count/scatter)
// and, after scatter restores it, as the final count for each cell.
@group(0) @binding(4) var<storage, read_write> cell_counts: array<atomic<u32>>;
// Exclusive prefix sum of cell_counts: start index of each cell's particles
// within `sorted_indices`.
@group(0) @binding(5) var<storage, read_write> cell_offsets: array<u32>;
// Particle indices grouped by cell (a counting-sort permutation of 0..N).
@group(0) @binding(6) var<storage, read_write> sorted_indices: array<u32>;

fn cell_coords(pos: vec2<f32>) -> vec2<u32> {
    let fx = max(pos.x + params.half_width, 0.0);
    let fy = max(pos.y + params.half_height, 0.0);
    let cx = min(u32(fx / params.cell_size), params.grid_cols - 1u);
    let cy = min(u32(fy / params.cell_size), params.grid_rows - 1u);
    return vec2<u32>(cx, cy);
}

fn cell_index(coords: vec2<u32>) -> u32 {
    return coords.y * params.grid_cols + coords.x;
}

@compute @workgroup_size(64)
fn clear_counts(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_cells) {
        return;
    }
    atomicStore(&cell_counts[i], 0u);
}

@compute @workgroup_size(64)
fn count_particles(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) {
        return;
    }
    let cell = cell_index(cell_coords(particles_in[i].pos));
    atomicAdd(&cell_counts[cell], 1u);
}

// Single-thread exclusive prefix sum. The grid is tiny (a few hundred cells
// at most, since cell size == max_radius), so a sequential scan here is
// negligible next to the O(num_particles) passes around it.
@compute @workgroup_size(1)
fn prefix_sum(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) {
        return;
    }
    var running: u32 = 0u;
    for (var i: u32 = 0u; i < params.num_cells; i = i + 1u) {
        cell_offsets[i] = running;
        running = running + atomicLoad(&cell_counts[i]);
    }
}

@compute @workgroup_size(64)
fn scatter_particles(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) {
        return;
    }
    let cell = cell_index(cell_coords(particles_in[i].pos));
    // cell_counts was reset to 0 (via clear_counts) before this pass, so this
    // atomicAdd both hands out this particle's local slot within the cell and
    // rebuilds cell_counts back to the true per-cell counts as a side effect.
    let local = atomicAdd(&cell_counts[cell], 1u);
    sorted_indices[cell_offsets[cell] + local] = i;
}

fn attraction(r: f32, a: f32, beta: f32) -> f32 {
    if (r < beta) {
        return r / beta - 1.0;
    } else if (r < 1.0) {
        return a * (1.0 - abs(2.0 * r - 1.0 - beta) / (1.0 - beta));
    }
    return 0.0;
}

@compute @workgroup_size(64)
fn compute_forces(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) {
        return;
    }

    let p = particles_in[i];
    var accel = vec2<f32>(0.0, 0.0);
    let world_w = params.half_width * 2.0;
    let world_h = params.half_height * 2.0;

    let home = cell_coords(p.pos);
    let cols = i32(params.grid_cols);
    let rows = i32(params.grid_rows);

    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            // Wrap neighbor cell coords toroidally, matching particle wrap.
            let ncx = u32((i32(home.x) + dx + cols) % cols);
            let ncy = u32((i32(home.y) + dy + rows) % rows);
            let cell = ncy * params.grid_cols + ncx;
            let start = cell_offsets[cell];
            let count = atomicLoad(&cell_counts[cell]);

            for (var k: u32 = 0u; k < count; k = k + 1u) {
                let j = sorted_indices[start + k];
                if (j == i) {
                    continue;
                }
                let o = particles_in[j];
                var d = o.pos - p.pos;
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
