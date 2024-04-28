#version 330 core

in float velocityTowardsObserver;
out vec4 FragColor;

void main() {
    float redShift = 0.0;
    float blueShift = 0.0;
    
    blueShift = velocityTowardsObserver;
    redShift = -velocityTowardsObserver;

    FragColor = vec4(redShift, 0.0, blueShift, 1.0);
}