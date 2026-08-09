struct FragmentInput {
    @location(0) uv: vec2<f32>,
    @location(1) color: vec3<f32>,
};

@fragment
fn main(in: FragmentInput) -> @location(0) vec4<f32> {
    let d = length(in.uv);
    if (d > 1.0) {
        discard;
    }
    let alpha = smoothstep(1.0, 0.7, d);
    return vec4<f32>(in.color, alpha);
}
