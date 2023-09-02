#version 330 core

layout(location = 0) in vec3 aPosition;

uniform mat4 viewProjection;

void main()
{
    gl_Position = viewProjection * vec4(aPosition, 1.0);
}