#! /usr/bin/env python3
import glfw
from OpenGL.GL import *
import pyopencl as cl
import numpy as np
from glm import *
import time
from pathlib import Path

num_particles = 1000000
window_width = 800
window_height = 600
change_attractor = False
scroll_offset = 0.0
cube_init_kernel = None
sphere_init_kernel = None

def reset_simulation(queue, particles_buffer, velocities_buffer, accelerations_buffer, sphere):
    if sphere:
        sphere_init_kernel.set_args(particles_buffer, velocities_buffer, accelerations_buffer, np.uint32(num_particles))
        cl.enqueue_nd_range_kernel(queue, sphere_init_kernel, (num_particles,), None)
    else:
        cube_init_kernel.set_args(particles_buffer, velocities_buffer, accelerations_buffer, np.uint32(num_particles))
        cl.enqueue_nd_range_kernel(queue, cube_init_kernel, (num_particles,), None)


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
        self.attractor = vec4(0.0,0.0,0.0,1.0)
        self.sphere = False
        self.fullscreen = False

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
        if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
            self.position -= normalize(cross(self.front, self.up)) * self.move_speed * delta_time
        if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
            self.position += normalize(cross(self.front, self.up)) * self.move_speed * delta_time
        if glfw.get_key(window, glfw.KEY_T) == glfw.PRESS:
            self.sphere = not self.sphere
            print(f"sphere {'on' if self.sphere else 'off'}")
            time.sleep(0.2)
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(window, True)
        if glfw.get_key(window, glfw.KEY_F) == glfw.PRESS:
            self.fullscreen = not self.fullscreen
            if self.fullscreen:
                monitor = glfw.get_primary_monitor()
                mode = glfw.get_video_mode(monitor)
                glfw.set_window_monitor(window, monitor, 0, 0, mode.size[0], mode.size[1], mode.refresh_rate)
            else:
                glfw.set_window_monitor(window, None, 0, 0, window_width, window_height, 0)

        return {
            'reset_simulation': glfw.get_key(window, glfw.KEY_R) == glfw.PRESS,
            'toggle_attractor': glfw.get_key(window, glfw.KEY_G) == glfw.PRESS,
            'toggle_attractor_status': glfw.get_key(window, glfw.KEY_O) == glfw.PRESS,
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
        self.pitch += yoffset

        if self.pitch > 89.0:
            self.pitch = 89.0
        if self.pitch < -89.0:
            self.pitch = -89.0

        direction = vec3()
        direction.x = cos(radians(self.yaw)) * cos(radians(self.pitch))
        direction.y = sin(radians(self.pitch))
        direction.z = sin(radians(self.yaw)) * cos(radians(self.pitch))
        self.front = normalize(direction)

    def get_view_matrix(self):
        return lookAt(self.position, self.position + self.front, self.up)

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

def framebuffer_size_callback(window, width, height):
    global window_width, window_height
    glViewport(0, 0, width, height)
    window_width = width
    window_height = height


def scroll_callback(window, x_off, y_off):
    global scroll_offset
    scroll_offset += y_off * 0.1
    if scroll_offset < 1: scroll_offset = 1


def main():
    global window_width, window_height, change_attractor, scroll_offset, cube_init_kernel, sphere_init_kernel

    if cl.have_gl():
        print("PyOpenCL has OpenGL support.")
    else:
        print("PyOpenCL does not have OpenGL support.")
    return

    if not glfw.init():
        return

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 0)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.VISIBLE, glfw.TRUE)

    window = glfw.create_window(window_width, window_height, "Particle System", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    glfw.set_window_title(window, "Particle System")
    glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)
    glfw.set_scroll_callback(window, scroll_callback)

    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE)

    doppler_vs = create_shader(GL_VERTEX_SHADER, Path("./shaders/doppler.vs").read_text())
    doppler_fs = create_shader(GL_FRAGMENT_SHADER, Path("./shaders/doppler.fs").read_text())
    doppler_program_id = create_program(doppler_vs, doppler_fs)

    doppler_pos_attrib = glGetAttribLocation(doppler_program_id, "aPosition")
    doppler_vel_attrib = glGetAttribLocation(doppler_program_id, "aVelocity")

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, num_particles * 4 * sizeof(GLfloat) * 2, None, GL_DYNAMIC_DRAW)

    attractor_vs = create_shader(GL_VERTEX_SHADER, Path("shaders/attractor.vs").read_text())
    attractor_fs = create_shader(GL_FRAGMENT_SHADER, Path("shaders/attractor.fs").read_text())
    attractor_program_id = create_program(attractor_vs, attractor_fs)

    platforms = cl.get_platforms()
    devices = platforms[0].get_devices()

    devices = [d for d in platforms[0].get_devices() if d.version >= "OpenCL 1.2"]
    if not devices:
        print("No devices support OpenCL 1.2")
        return

    cl_ctx = cl.Context(devices)

    queue = cl.CommandQueue(cl_ctx)

    cl_particles_buffer = cl.Buffer(cl_ctx, cl.mem_flags.READ_WRITE, int(vbo))

    cl_program = cl.Program(cl_ctx, Path("shaders/kernel.compute").read_text()).build()
    update_kernel = cl_program.update_particles
    sphere_init_kernel = cl_program.initialize_particles_sphere
    cube_init_kernel = cl_program.initialize_particles_cube

    camera = Camera()

    glfw.show_window(window)
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)

    mode = glfw.get_video_mode(glfw.get_primary_monitor())
    projection = perspective(camera.fov, mode.size[0] / mode.size[1], camera.near, camera.far)
    glEnable(GL_DEPTH_TEST)

    last_frame = glfw.get_time()
    while not glfw.window_should_close(window):
        current_frame = glfw.get_time()
        delta_time = current_frame - last_frame
        last_frame = current_frame

        input_flags = camera.keyboard_input(window, delta_time)

        if input_flags.get('reset_simulation'):
            reset_simulation(queue, cl_particles_buffer, camera.sphere)
            update_kernel.set_args(cl_particles_buffer, camera.attractor, np.float32(0.01), np.uint32(num_particles))

        if input_flags.get('toggle_attractor'):
            change_attractor = not change_attractor
            scroll_offset = 3.0

        if input_flags.get('toggle_attractor_status'):
            camera.attractor.w = not camera.attractor.w
            print(f"attractor {'on' if camera.attractor.w else 'off'}")

        if change_attractor or input_flags.get('toggle_attractor_status'):
            update_kernel.set_args(cl_particles_buffer, camera.attractor, np.float32(0.01), np.uint32(num_particles))

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if change_attractor:
            view = camera.get_view_matrix()
            view_projection = projection * view
            glUseProgram(attractor_program_id)
            glUniformMatrix4fv(glGetUniformLocation(attractor_program_id, "viewProjection"), 1, GL_FALSE, value_ptr(view_projection))
            draw_sphere_surface(camera.attractor.xyz, 0.05, 10)
            camera.attractor = vec4(camera.position + camera.front * scroll_offset, camera.attractor.w)
            update_kernel.set_args(cl_particles_buffer, camera.attractor, np.float32(0.01), np.uint32(num_particles))

        glFinish()
        cl.enqueue_acquire_gl_objects(queue, [cl_particles_buffer])
        cl.enqueue_nd_range_kernel(queue, update_kernel, (num_particles,), None)
        cl.enqueue_release_gl_objects(queue, [cl_particles_buffer])
        queue.finish()

        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glVertexAttribPointer(doppler_pos_attrib, 4, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(doppler_pos_attrib)
        glVertexAttribPointer(doppler_vel_attrib, 4, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(doppler_vel_attrib)
        glDrawArrays(GL_POINTS, 0, num_particles)

        fps = 1.0 / delta_time
        fps_interval += delta_time
        if fps_interval > 0.5:
            glfw.set_window_title(window, f"Particle System | FPS: {fps:.2f}")
            fps_interval = 0

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()