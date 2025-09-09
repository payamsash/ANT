#ifdef GL_ES
    precision heighp float;
#endif

uniform vec2 u_resolution;
uniform float u_time;
uniform float u_alpha_signal; 
uniform vec3 u_color;
uniform vec3 u_backgroundColor;

const float N = 20.0;

float mapRange(float value, float oldMin, float oldMax, float newMin, float newMax) {
    return newMin + (value - oldMin) / (oldMax - oldMin) * (newMax - newMin);
}

void main() {
    // Calculate normalized pixel coordinates (from -1 to 1)
    vec2 u = (2.0 * gl_FragCoord.xy - u_resolution.xy) / u_resolution.y;

    float scale = mapRange(u_alpha_signal, 8.0, 13.0, 1.0, 8.0);
    // Apply the scaling to p to scale the drawing
    u *= scale;
   float dynamic_N = mapRange(u_alpha_signal, 8.0, 13.0, N, 80.0);

    // Initialize time, radius, and angle
    float t = u_time * 5.0;
    float r = length(u);      // Radius
    float a = atan(u.y, u.x); // Angle in polar coordinates

    // Calculate index based on radius and adjust angle and radius
    float i = floor(r * N);
    a *= floor(pow(128.0, i / N));
    a += 10.0 * t + 123.34 * i;

    r += (0.5 + 0.5 * cos(a)) / dynamic_N;
    r = floor(N * r) / N;

    // Set background color
    vec4 backgroundColor = vec4(u_backgroundColor, 1.0);
    // Calculate the color with increased intensity for the outside
    vec4 color = (1.0 - r) * vec4(u_color * 0.75, 1.0); // Increase intensity
    color.rgb = clamp(u_color.rgb, 0.0, 1.0); // Ensure color values remain valid
    // Combine background and ink color with smooth transition
    float blendFactor = smoothstep(0.05, 1.0, 1.0 - r);
    gl_FragColor = mix(backgroundColor, color, blendFactor);
}
