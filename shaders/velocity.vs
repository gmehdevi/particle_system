#version 330 core

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aVelocity;

uniform mat4 viewProjection;

out float speed;

void main() {
    gl_Position = viewProjection * vec4(aPosition, 1.0);
    speed = length(aVelocity); 
}
