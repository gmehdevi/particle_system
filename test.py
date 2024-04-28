import pyopencl as cl
from pyopencl.tools import get_gl_sharing_context_properties
from OpenGL.GL import *
import glfw

def main():
    if not glfw.init():
        print("Failed to initialize GLFW")
        return

    # Create an invisible window to establish an OpenGL context
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(640, 480, "Invisible Window", None, None)
    if not window:
        print("Failed to create GLFW window")
        glfw.terminate()
        return

    glfw.make_context_current(window)

    try:
        # Get context properties for CL-GL sharing
        context_properties = get_gl_sharing_context_properties()
        # Add platform and device type to properties
        platform = cl.get_platforms()[0]
        devices = platform.get_devices()

        # Filter to the device which can share with OpenGL context
        context = None
        for device in devices:
            try:
                context = cl.Context(properties=context_properties + [(cl.context_properties.PLATFORM, platform),
                                                                      (cl.context_properties.GL_CONTEXT_KHR, glfw.get_current_context())],
                                     devices=[device])
                print(f"Successfully created a shared context with device: {device.name}")
                break
            except cl.LogicError as e:
                print(f"Failed to create a shared context with device: {device.name}, error: {str(e)}")

        if context is None:
            print("No compatible device found for CL-GL sharing.")
            return

        # Create an OpenGL buffer
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        data = (GLfloat * 1)(0)  # Just dummy data
        glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW)

        # Create OpenCL buffer from OpenGL buffer
        cl_buffer = cl.GLBuffer(context, cl.mem_flags.READ_WRITE, int(vbo))

        print("CL-GL interoperability is supported on your setup!")
        print(f"OpenGL Buffer (VBO) ID: {vbo}")
        print(f"OpenCL Buffer (from VBO) ID: {cl_buffer.int_ptr}")

    except Exception as e:
        print(f"Failed to establish CL-GL interoperability: {str(e)}")

    finally:
        glfw.destroy_window(window)
        glfw.terminate()

if __name__ == "__main__":
    main()
