//! The on-screen control panel: layout, hit-testing, and the colored-triangle
//! geometry it's drawn with. Self-contained - no GPU handles, no Model - just
//! math in and a Vec of vertices out.

use nannou::prelude::*;

// Panel layout, in screen pixels.
const UI_MARGIN: f32 = 16.0;
const UI_CELL: f32 = 22.0;
const UI_GAP: f32 = 3.0;
const UI_SLIDER_W: f32 = 200.0;
const UI_SLIDER_H: f32 = 10.0;
const UI_ROW_GAP: f32 = 18.0;
pub const UI_MAX_VERTICES: usize = 1024;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UiVertex {
    pub clip_pos: [f32; 2],
    pub color: [f32; 4],
}

#[derive(Clone, Copy)]
pub struct UiRect {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

impl UiRect {
    pub fn contains(&self, p: Vec2) -> bool {
        p.x >= self.x0 && p.x <= self.x1 && p.y >= self.y0 && p.y <= self.y1
    }
}

pub struct UiLayout {
    pub slider_hit: UiRect,
    pub slider_track: UiRect,
    pub slider_x0: f32,
    pub slider_x1: f32,
    pub slider_y_center: f32,
    pub grid_origin: (f32, f32),
    pub panel_bg: UiRect,
}

pub fn compute_ui_layout(half_w: f32, half_h: f32, num_types: u32) -> UiLayout {
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
pub fn grid_cell_rect(origin: (f32, f32), row: u32, col: u32) -> UiRect {
    let x0 = origin.0 + col as f32 * (UI_CELL + UI_GAP);
    let y1 = origin.1 - row as f32 * (UI_CELL + UI_GAP);
    UiRect {
        x0,
        x1: x0 + UI_CELL,
        y0: y1 - UI_CELL,
        y1,
    }
}

pub fn push_triangle(
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
pub fn push_down_chevron(verts: &mut Vec<UiVertex>, rect: &UiRect, color: [f32; 4], half_w: f32, half_h: f32) {
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
pub fn push_right_chevron(verts: &mut Vec<UiVertex>, rect: &UiRect, color: [f32; 4], half_w: f32, half_h: f32) {
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

pub fn push_rect(verts: &mut Vec<UiVertex>, rect: &UiRect, color: [f32; 4], half_w: f32, half_h: f32) {
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

pub fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
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
pub fn type_color_rgb(t: u32) -> [f32; 4] {
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
pub fn matrix_value_color(v: f32) -> [f32; 4] {
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
