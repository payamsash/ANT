#ifdef GL_ES
precision mediump float;
#endif

uniform vec2 u_resolution; // Screen resolution (set in the main application)
uniform float u_time;      // Time (in seconds, set in the main application)
uniform float u_alpha_signal;
uniform vec3 u_inkColor1;
uniform vec3 u_paperColor;

float mapRange(float value, float oldMin, float oldMax, float newMin, float newMax) {
    return newMin + (value - oldMin) / (oldMax - oldMin) * (newMax - newMin);
}

void mainImage(out vec4 O, vec2 U)
{
    // Initialize resolution and normalize coordinates
    vec2 R = u_resolution;
    U = (U + U - R) / R.y;
    float scale = mapRange(u_alpha_signal, 8.0, 13.0, 2.0, 0.8);
    // Apply the scaling to p to scale the drawing
    U *= scale;
    // Initialize polar coordinates and output color
    float a = atan(U.y, U.x);
    float r = length(U);
    float A, d;

    // Map the alpha signal to a suitable speed factor for curve calculations
    float fratcionalFactor = mapRange(u_alpha_signal, 8.0, 13.0, 9.0, 7.0);
    float speedFactor = mapRange(u_alpha_signal, 8.0, 13.0, 3.0, 1.5);
    float pointFactor = mapRange(u_alpha_signal, 8.0, 13.0, 50.0, 100.0);
    float diskFactor = mapRange(u_alpha_signal, 8.0, 13.0, 0.5, 0.1);

    // Set paper color as the background
    O = vec4(u_paperColor, 1.0);

    // Loop for creating the nested curves
    vec3 color = vec3(0.0); // Initialize color accumulator
    for (int i = 0; i < 3; i++) {
        // Calculate curve parameters using speedFactor
        A = fratcionalFactor / 3.0 * a + u_time * speedFactor; 
        R = vec2(fract(a * pointFactor / 6.283) - 0.5, 16.0 * (r - 0.2 * sin(A) - 0.5)); // Texture param

        // Calculate small disks and modulate
        d = smoothstep(diskFactor, 0.0, length(R));
        color += (1.0 + cos(A)) * d * u_inkColor1; // Accumulate color using ink color

        a += 6.283; // Move to next nested curve
    }
    // Blend the accumulated ink color with the background paper color
    float blendFactor = clamp(length(color) / 2.0, 0.0, 1.0);
    O.rgb = mix(u_paperColor, color, blendFactor);
    O.a = 1.0; // Set alpha to fully opaque
}

void main() {
    vec4 color;
    mainImage(color, gl_FragCoord.xy);
    gl_FragColor = color;
}
