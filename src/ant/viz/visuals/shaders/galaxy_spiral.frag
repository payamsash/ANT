#ifdef GL_ES
precision highp float;
#endif

uniform vec2 u_resolution;       // Screen resolution
uniform float u_time;            // Time in seconds
uniform vec3 u_color;        //  color
uniform vec3 u_backgroundColor;       // background color
uniform float u_alpha_signal;


#define rot(a) mat2(cos(a), -sin(a), sin(a), cos(a))

// Simple hash function for random number generation
float hash(vec2 p) {
    p = mod(p, 289.0);
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

// 2D noise function with smooth interpolation
float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Replaces the texture function with procedural noise
vec4 tex(vec2 U) {
    float T = noise(U * 250.0);
    float n = 1.0 - abs(2.0 * T - 1.0);
    return vec4(n, n, n, 1.0);
}

// Procedural pattern function
vec4 fractalTex(vec2 U) {
    U /= 12.0;
    return tex(U) * tex(2.0 * U) * 1.5 * tex(4.0 * U) * 1.5 * tex(8.0 * U) * 1.5;
}

// Map range utility
float mapRange(float value, float oldMin, float oldMax, float newMin, float newMax) {
    return newMin + (value - oldMin) / (oldMax - oldMin) * (newMax - newMin);
}

void main() {
    vec2 u = (2.0 * gl_FragCoord.xy - u_resolution.xy) / u_resolution.y;
    float scaleMod = 1.0 + 0.2 * sin(u_time * 0.3);
    float scale = mapRange(u_alpha_signal, 8.0, 13.0, 3.5, 15.0) * scaleMod;
    float speed = mapRange(u_alpha_signal, 8.0, 13.0, 5.0, 15.0);
    // Apply the scaling to p to scale the drawing
    u *= scale;
    vec4 O = vec4(0.0);

    vec2 r = vec2(0.6, 0.3);         // Base ellipse aspect ratio
    float angleStep  = -3.14 / 2.0;           // Ellipse angle tilt per scaling length
    float thickness  = 0.4;                   // Thickness of each ellipse

    for (float l = 1.0; l < 3.5; l += 0.1) {
        vec2 V = 1.0 / r * (rot(angleStep + angleStep * l) * u);
        float d = dot(V, V);
        float va = fract(u_time *0.1) * 6.283185 * (speed / l);
        vec4 C = fractalTex(rot(va + l) * 0.6 * V / l);
        O += smoothstep(thickness, 0.0, abs(sqrt(d) - l)) * C / l;
    }
    // Map the intensity to ink and paper colors
    vec3 color = mix(u_backgroundColor, u_color, O.r); // Blend based on intensity
    gl_FragColor = vec4(color * exp(-O.r / 8.0), 1.0); // Apply exponential decay to the color
}
