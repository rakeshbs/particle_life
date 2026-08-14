//! Interaction-matrix presets. All built around asymmetric attraction — a
//! type chases another type which only weakly (or doesn't) chase back. That
//! asymmetry is what makes a bonded group self-propel and drift across the
//! screen instead of settling into a static cluster, so every preset here
//! produces gliders of one flavor or another.

pub const PRESET_COUNT: usize = 8;

fn pair_partner(a: u32) -> u32 {
    if a % 2 == 0 { a + 1 } else { a - 1 }
}

// Three independent pairs (0-1, 2-3, 4-5), strong chase / strong flee: small,
// fast-moving gliders.
fn fast_pairs(n: u32) -> Vec<f32> {
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
fn slow_pairs(n: u32) -> Vec<f32> {
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
fn cohesive_gliders(n: u32) -> Vec<f32> {
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
fn tight_darts(n: u32) -> Vec<f32> {
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
fn triad_chasers(n: u32) -> Vec<f32> {
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
fn worm_train(n: u32) -> Vec<f32> {
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
fn twin_worms(n: u32) -> Vec<f32> {
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
fn swarm_chase(n: u32) -> Vec<f32> {
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

pub fn preset_matrix(index: usize, num_types: u32) -> Vec<f32> {
    match index % PRESET_COUNT {
        0 => fast_pairs(num_types),
        1 => slow_pairs(num_types),
        2 => cohesive_gliders(num_types),
        3 => tight_darts(num_types),
        4 => triad_chasers(num_types),
        5 => worm_train(num_types),
        6 => twin_worms(num_types),
        _ => swarm_chase(num_types),
    }
}
