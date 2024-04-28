#version 330 core

in float speed;
out vec4 FragColor;

void main() {
    vec3 color = mix(vec3(0.5, 0.5, 1.0), vec3(1.0, 0.5, 0.5), clamp(speed / 5.0, 0.0, 1.0));
    FragColor = vec4(color, 1.0);
}
