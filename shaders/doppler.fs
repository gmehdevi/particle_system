#version 330 core
out vec4 FragColor;

in float redShift;
in float blueShift;

void main()
{
    FragColor = vec4(max(0.5, 0.5 + redShift), 0.25, max(0.5, 0.5 + blueShift), 1.0);
}