#version 330 core

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aVelocity;

uniform mat4 viewProjection;
uniform vec3 cameraPosition;

out float velocityTowardsObserver;

void main() {
    gl_Position = viewProjection * vec4(aPosition, 1.0);

    vec3 velocityDirection = normalize(aVelocity);
    vec3 toCamera = normalize(cameraPosition - aPosition);

    velocityTowardsObserver = dot(velocityDirection, toCamera);
}