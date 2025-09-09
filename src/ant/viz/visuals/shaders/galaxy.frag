#ifdef GL_ES
    precision heighp float;
#endif

uniform vec2 u_resolution;      // Screen resolution
uniform float u_time;           // Time (in seconds)
uniform float u_alpha_signal;   // Alpha signal for customization
uniform vec3 u_color;           // Star color (should be set to blue)
uniform float u_speed;          // Speed parameter

const float uStarSize = 10.0;

float mapRange(float value, float oldMin, float oldMax, float newMin, float newMax) {
    return newMin + (value - oldMin) / (oldMax - oldMin) * (newMax - newMin);
}

float rand(vec2 co) {
    return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

float randInRange(vec2 co, float min, float max) {
    return min + (max - min) * fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

float makeStar(vec2 fragCoord, float dist, float speed, float size, vec2 seed) {
    vec2 currPos = u_resolution * 0.5;
    currPos.x += dist * sin(u_time * speed * 0.5);
    currPos.y += dist * cos(u_time * speed * 0.5);
    return size / (1.0 + length(currPos - fragCoord) * 20.0);
}

void main() {
    vec2 fragCoord = gl_FragCoord.xy;

    // Map the alpha signal to a suitable speed factor for curve calculations
    float uDistanceFactor = mapRange(u_alpha_signal, 8.0, 13.0, 400.0, 50.0);
    float uNumStars = mapRange(u_alpha_signal, 8.0, 13.0, 125.0, 30.0);

    vec3 backgroundColor = vec3(1.0); // White background
    vec3 finalColor = backgroundColor; // Initialize with the background color
    vec2 derpseed = vec2(3.44, 9.12);

    for (int i = 0; i < uNumStars; i++) {
        vec3 starColor = u_color; // Star color
        float dist = rand(derpseed) * uDistanceFactor;
        float speed = rand(derpseed * 3.99) * u_speed;
        float size = rand(derpseed * 225.22) * uStarSize;
        float starIntensity = makeStar(fragCoord, dist, speed, size, derpseed) * randInRange(derpseed, 0.5, 1.0);

        // Blend star color over the background
        finalColor = mix(finalColor, starColor, starIntensity);

        derpseed.x += 0.75;  // Increase the value to create more spread
        derpseed.y += 0.37;  // Increase the value to create more spread
    }

    gl_FragColor = vec4(finalColor, 1.0);
}
