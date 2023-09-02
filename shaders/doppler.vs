#version 330 core

layout(location = 0) in vec4 aPosition;
layout(location = 1) in vec4 aVelocity;

uniform mat4 viewProjection;
uniform vec3 cameraPosition;

out float redShift;
out float blueShift;

void main()
{
    gl_Position = viewProjection * vec4(aPosition.xyz, 1.0);
    gl_PointSize = 2.0 / length(cameraPosition - aPosition.xyz);

    vec3 toParticle = aPosition.xyz - cameraPosition;
    float distance = length(toParticle);
    
    vec3 normalizedVelocity = normalize(aVelocity.xyz);
    float relativeVelocity = dot(normalizedVelocity, toParticle);
    float dopplerShift = 0.5 / (0.5 - relativeVelocity * 0.5);

    blueShift = distance / dopplerShift;
    redShift = distance * dopplerShift;
}