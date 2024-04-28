from OpenGL.GL import *
import glfw
import pyopencl as cl
from pyopencl.tools import get_gl_sharing_context_properties
import numpy as np
from glm import *
from pathlib import Path
import time
import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

n_particles = 10000000
window_width, window_height = 800, 800


def reshape(window, width, height):
    glViewport(0, 0, width, height)

def framebuffer_size_callback(window, width, height):
    global window_width, window_height
    glViewport(0, 0, width, height)
    window_width = width
    window_height = height

def reset_simulation(queue, pos, vel, acc, sphere, prog, n_particles):
    cl.enqueue_acquire_gl_objects(queue, [pos, vel, acc])

    if sphere:
        prog.initialize_particles_sphere(queue, (n_particles,), None, pos, vel, acc, np.uint32(n_particles), np.float32(0.5))
    else:
        prog.initialize_particles_cube(queue, (n_particles,), None, pos, vel, acc, np.uint32(n_particles))

    cl.enqueue_release_gl_objects(queue, [pos, vel, acc])
    queue.finish()


def create_shader(shader_type, shader_source):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, shader_source)
    glCompileShader(shader)
    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(shader))
    return shader

def create_program(vertex_shader, fragment_shader):
    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)
    if glGetProgramiv(program, GL_LINK_STATUS) != GL_TRUE:
        raise RuntimeError(glGetProgramInfoLog(program))
    return program

def draw_sphere_surface(position, radius, num_segments):
    for j in range(num_segments):
        lat0 = np.pi * (-0.5 + (j - 1) / num_segments)
        z0 = radius * np.sin(lat0)
        zr0 = radius * np.cos(lat0)

        lat1 = np.pi * (-0.5 + j / num_segments)
        z1 = radius * np.sin(lat1)
        zr1 = radius * np.cos(lat1)

        glBegin(GL_TRIANGLE_STRIP)
        for i in range(num_segments + 1):
            lng = 2 * np.pi * (i - 1) / num_segments
            x = np.cos(lng)
            y = np.sin(lng)

            glNormal3f(position.x + x * zr0, position.y + y * zr0, position.z + z0)
            glVertex3f(position.x + x * zr0, position.y + y * zr0, position.z + z0)
            glNormal3f(position.x + x * zr1, position.y + y * zr1, position.z + z1)
            glVertex3f(position.x + x * zr1, position.y + y * zr1, position.z + z1)
        glEnd()

class Camera:
    def __init__(self):
        self.position = vec3(0.0, 0.0, 3.0)
        self.front = -normalize(self.position)
        self.up = vec3(0.0, 1.0, 0.0)
        self.yaw = -90.0
        self.pitch = 0.0
        self.move_speed = 2.5
        self.mouse_sensitivity = 10
        self.last_x = window_width / 2
        self.last_y = window_height / 2
        self.first_mouse = True
        self.near = 0.1
        self.far = 1000000
        self.fov = radians(45)
        self.fullscreen = False
        self.scroll_offset = 1.0
        self.window_width = window_width
        self.window_height = window_height


    def keyboard_input(self, window, delta_time):
        vertical_movement = 0.0

        if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS:
            vertical_movement += self.move_speed
        if glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS:
            vertical_movement -= self.move_speed
        self.position += self.up * vertical_movement * delta_time

        if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
            self.position += self.front * self.move_speed * delta_time
        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
            self.position -= self.front * self.move_speed * delta_time
        if glfw.get_key(window, glfw.KEY_A) is glfw.PRESS:
            self.position -= normalize(cross(self.front, self.up)) * self.move_speed * delta_time
        if glfw.get_key(window, glfw.KEY_D) is glfw.PRESS:
            self.position += normalize(cross(self.front, self.up)) * self.move_speed * delta_time

        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(window, True)
            glfw.terminate()
            os._exit(0)
        elif glfw.get_key(window, glfw.KEY_F) == glfw.PRESS:
            if self.fullscreen:
                monitor = glfw.get_primary_monitor()
                mode = glfw.get_video_mode(monitor)
                glfw.set_window_monitor(window, monitor, 0, 0, mode.size[0], mode.size[1], mode.refresh_rate)
                self.window_width, self.window_height = mode.size[0], mode.size[1]
            else:
                glfw.set_window_monitor(window, None, 50, 50, window_width, window_height, 0)
                self.window_width, self.window_height = window_width, window_height
            self.fullscreen = not self.fullscreen
            time.sleep(0.1)
            
        return {
            'reset': glfw.get_key(window, glfw.KEY_R) == glfw.PRESS,
            'change_attractor': glfw.get_key(window, glfw.KEY_C) == glfw.PRESS,
            'sphere' : glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS,
            'toggle_attractor' : glfw.get_key(window, glfw.KEY_T) == glfw.PRESS,
            'velocity' : glfw.get_key(window, glfw.KEY_V) == glfw.PRESS
        }

    def mouse_input(self, window, delta_time):
        xpos, ypos = glfw.get_cursor_pos(window)
        if self.first_mouse:
            self.last_x = xpos
            self.last_y = ypos
            self.first_mouse = False

        xoffset = xpos - self.last_x
        yoffset = self.last_y - ypos
        self.last_x = xpos
        self.last_y = ypos

        xoffset *= self.mouse_sensitivity * delta_time
        yoffset *= self.mouse_sensitivity * delta_time

        self.yaw += xoffset
        self.pitch = max(min(self.pitch + yoffset, 89.0), -89.0)

        direction = vec3(cos(radians(self.yaw)) * cos(radians(self.pitch)),
                         sin(radians(self.pitch)),
                         sin(radians(self.yaw)) * cos(radians(self.pitch)))
        self.front = normalize(direction)

    def get_view_matrix(self):
        return lookAt(self.position, self.position + self.front, self.up)

    def get_projection_matrix(self):
        return perspective(self.fov, float(self.window_width) / self.window_height, self.near, self.far)

    def scroll_callback(self, window, x_off, y_off):
        self.scroll_offset += y_off * 0.1
        if self.scroll_offset < 1: self.scroll_offset = 1

    
def create_buffers(ctx, n_particles):
    buffer_size = n_particles * 4 * 4

    pos_vbo_id = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, pos_vbo_id)
    glBufferData(GL_ARRAY_BUFFER, buffer_size, None, GL_STATIC_DRAW)
    pos_vbo = cl.GLBuffer(ctx, cl.mem_flags.READ_WRITE, pos_vbo_id)

    vel_vbo_id = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vel_vbo_id)
    glBufferData(GL_ARRAY_BUFFER, buffer_size, None, GL_STATIC_DRAW)
    vel_vbo = cl.GLBuffer(ctx, cl.mem_flags.READ_WRITE, vel_vbo_id)
    
    acc_vbo_id = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, acc_vbo_id)
    glBufferData(GL_ARRAY_BUFFER, buffer_size, None, GL_STATIC_DRAW)
    acc_vbo = cl.GLBuffer(ctx, cl.mem_flags.READ_WRITE, acc_vbo_id)
    
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    return pos_vbo, vel_vbo, acc_vbo, pos_vbo_id, vel_vbo_id
   
    
def initialize():
    platform = cl.get_platforms()[0]
    ctx_properties = get_gl_sharing_context_properties()
    ctx = cl.Context(properties=ctx_properties, devices=[platform.get_devices()[0]])
    pos, vel, acc, pos_vbo, vel_vbo  = create_buffers(ctx, n_particles)
    prog = cl.Program(ctx, 
                      Path('shaders/kernel.compute').read_text()).build()
    queue = cl.CommandQueue(ctx)
    reset_simulation(queue, pos, vel, acc, True, prog, n_particles)
    queue.finish()
    return pos_vbo, vel_vbo, pos, vel, acc, queue, prog

def main_loop(window, pos_vbo, vel_vbo, pos, vel, acc, queue, prog, camera):
    sphere = True
    change_attractor = False
    velocity = False
    
    attractor = vec4(0.0, 0.0, 0.0, 1.0)
    vertex_shader = create_shader(GL_VERTEX_SHADER, 
                                  Path('shaders/doppler.vs').read_text())
    fragment_shader = create_shader(GL_FRAGMENT_SHADER,
                                    Path('shaders/doppler.fs').read_text())
    shader_program = create_program(vertex_shader, fragment_shader)
    
    vertex_shader = create_shader(GL_VERTEX_SHADER, 
                                  Path('shaders/attractor.vs').read_text())
    fragment_shader = create_shader(GL_FRAGMENT_SHADER, 
                                    Path('shaders/attractor.fs').read_text())
    att_shader_program = create_program(vertex_shader, fragment_shader)
    
    vertex_shader = create_shader(GL_VERTEX_SHADER,
                                  Path('shaders/velocity.vs').read_text())
    fragment_shader = create_shader(GL_FRAGMENT_SHADER,
                                    Path('shaders/velocity.fs').read_text())
    velocity_shader_program = create_program(vertex_shader, fragment_shader)
        
    glEnable(GL_BLEND)
    glEnable(GL_DEPTH_TEST)
    last_frame = glfw.get_time()

    while not glfw.window_should_close(window):
        current_frame = glfw.get_time()
        dt = current_frame - last_frame
        last_frame = current_frame


        sleep = True
        glfw.poll_events()
        inputs = camera.keyboard_input(window, dt)
        if inputs['reset']:
            reset_simulation(queue, pos, vel, acc, sphere, prog, n_particles)
        elif inputs['sphere']:
            sphere = not sphere
        elif inputs['change_attractor']:
            change_attractor = not change_attractor
        elif inputs['toggle_attractor']:
            attractor.w = 1.0 - attractor.w
        elif inputs['velocity']:
            velocity = not velocity
        else:
            sleep = False
        
        if sleep:
            time.sleep(0.1)            
            glfw.wait_events_timeout(2)
            glfw.poll_events()
            
            sleep = False
        camera.mouse_input(window, dt)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glUseProgram(att_shader_program)
        draw_sphere_surface(attractor.xyz, 0.01, 10)
        
        projection = camera.get_projection_matrix()
        view = camera.get_view_matrix();
        view_projection = projection * view;

        
        if velocity:
            glUseProgram(velocity_shader_program)
            vp_location = glGetUniformLocation(velocity_shader_program, "viewProjection")
            glUniformMatrix4fv(vp_location, 1, GL_FALSE, value_ptr(view_projection))
        else:
            glUseProgram(shader_program)
            vp_location = glGetUniformLocation(shader_program, "viewProjection")
            glUniformMatrix4fv(vp_location, 1, GL_FALSE, value_ptr(view_projection))
            
            cam_pos_location = glGetUniformLocation(shader_program, "cameraPosition")
            glUniform3f(cam_pos_location, camera.position.x, camera.position.y, camera.position.z)


        cl.enqueue_acquire_gl_objects(queue, [pos, vel, acc])
        prog.update_particles(queue, (n_particles,), None, pos, vel, acc, np.uint32(n_particles), np.float32(dt))
        if attractor.w == 1.0:
            prog.attractor(queue, (n_particles,), None, pos, vel, acc, attractor, np.uint32(n_particles), np.float32(dt))
        cl.enqueue_release_gl_objects(queue, [pos, vel, acc])
        queue.finish()
        
        
        glBindBuffer(GL_ARRAY_BUFFER, pos_vbo)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 16, None)

        glBindBuffer(GL_ARRAY_BUFFER, vel_vbo)
        glEnableVertexAttribArray(1) 
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 16, None)

        glDrawArrays(GL_POINTS, 0, n_particles)
        
        if change_attractor:
            attractor.xyz = camera.position + camera.front * camera.scroll_offset

        glUseProgram(att_shader_program)
        att_vp_location = glGetUniformLocation(att_shader_program, "viewProjection")
        glUniformMatrix4fv(att_vp_location, 1, GL_FALSE, value_ptr(view_projection))
        draw_sphere_surface(attractor.xyz, 0.01, 5)
        

        fps = 1.0 / dt
        glfw.set_window_title(window, f"Particle System | FPS: {fps:.2f}")
        
        glfw.swap_buffers(window)


if __name__ == '__main__':
    if not glfw.init():
        exit()
    window = glfw.create_window(window_width, window_height, "3D Sphere Simulation", None, None)
    glfw.window_hint(glfw.DECORATED, True)
    if not window:
        glfw.terminate()
        exit()
    camera = Camera()
    glfw.make_context_current(window)
    glfw.set_framebuffer_size_callback(window, reshape)
    glfw.set_scroll_callback(window, camera.scroll_callback)
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
    pos_vbo, vel_vbo, pos, vel, acc, queue, prog = initialize()
    main_loop(window, pos_vbo, vel_vbo, pos, vel, acc, queue, prog, camera)
    glfw.terminate()
