import  two_body as tb
import numpy as np
import math
from glumpy import app, gloo, gl

tb.init_data()
tb.compute_data()


n = np.sqrt(tb.hi_2) * pow(tb.a,-3/2)
print("n {0}".format(n))
β = tb.e/(1+np.sqrt(1-tb.e*tb.e))
mdle = np.sqrt((1+tb.e)/(1-tb.e))
E =  np.arctan(np.tan(tb.θ/2)/mdle)*2
print("E {0}".format(E))
M = E - tb.e* np.sin(E)
print("M {0}".format(M))
T = M/n
print("T {0}".format(T))

tb.b1.r = np.dot(tb.A, np.transpose(tb.b1.r))
tb.b2.r  = np.dot(tb.A, np.transpose(tb.b2.r))
tb.bc.r  = np.dot(tb.A, np.transpose(tb.bc.r))
tb.bc.v = np.dot(tb.A, np.transpose(tb.bc.v))
#print("E {0}".format(E))
time = 0
def compute_traectory(time):
    global E, M, T, n
    M = n*(time - T)
    E = M + tb.e * np.sin(E)

    x = tb.a * (np.cos(E) - tb.e)
    y = tb. a * np.sqrt(1- tb.e*tb.e) * np.sin(E)

    r0 = np.array([x,y,0],float)
    rc = tb.bc.r + tb.bc.v * time
    tb.b1.r = rc- r0 * (tb.b2.m/ tb.bc.m)
    tb.b2.r = rc + r0 * (tb.b1.m / tb.bc.m)
    return [(tb.b1.r[0],tb.b1.r[1]),(tb.b2.r[0],tb.b2.r[1])]

def compute_traectory_rk():
    """
    
    :param time: 
    :return:

    """
    pass


vertex = """
  attribute vec2 position;
  uniform float scale;
  void main()
  {
    gl_Position = vec4(scale*position, 0.0, 1.0);
    gl_PointSize = 5.0;
  } """

fragment = """
  varying vec4 v_color;
  void main()
  {
      gl_FragColor = vec4(vec3(0.0), 1.0);
  } """

# Build the program and corresponding buffers (with 4 vertices)
quad = gloo.Program(vertex, fragment, count=2)

# Upload data into GPU

quad['position'] = [(tb.b1.r[0], tb.b1.r[1]),(tb.b2.r[0], tb.b2.r[1])]
quad['scale'] = 0.08

# Create a window with a valid GL context
window = app.Window(color=(1,1,1,1))


# Tell glumpy what needs to be done at each redraw
@window.event

def on_draw(dt):
    global time
    window.clear()
    #quad["scale"] = math.cos(time)
    quad['position'] = compute_traectory(time)
    time += 0.01
    quad.draw(gl.GL_POINTS)

# We set the framecount to 360 in order to record a movie that can
# loop indefinetly. Run this program with:
# python quad-scale.py --record quad-scale.mp4
app.run()




"""

#import OpenGL.GL as gl
#import OpenGL.GLUT as glut
def display():
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
    glut.glutSwapBuffers()

def reshape(width,height):
    gl.glViewport(0, 0, width, height)

def keyboard( key, x, y ):
    if key == b'\x1b':
        sys.exit( )

glut.glutInit()
glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGBA)
glut.glutCreateWindow('Hello world!')
glut.glutReshapeWindow(512,512)
glut.glutReshapeFunc(reshape)
glut.glutDisplayFunc(display)
glut.glutKeyboardFunc(keyboard)
glut.glutMainLoop()


program  = gl.glCreateProgram()
vertex   = gl.glCreateShader(gl.GL_VERTEX_SHADER)
fragment = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)

vertex_code = "attribute vec2 position;" \
              "void main() " \
              "{ gl_Position = vec4(position, 0.0, 1.0); " \
              "" \
              "" \
              "}"



fragment_code = "void main() " \
                "{ gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0); " \
                "" \
                "}"

# Set shaders source
gl.glShaderSource(vertex, vertex_code)
gl.glShaderSource(fragment, fragment_code)

# Compile shaders
gl.glCompileShader(vertex)
if not gl.glGetShaderiv(vertex, gl.GL_COMPILE_STATUS):
    error = gl.glGetShaderInfoLog(vertex).decode()
    print(error)
    raise RuntimeError("Vertex shader compilation error")

gl.glCompileShader(fragment)
if not gl.glGetShaderiv(fragment, gl.GL_COMPILE_STATUS):
    error = gl.glGetShaderInfoLog(fragment).decode()
    print(error)
    raise RuntimeError("Fragment shader compilation error")

gl.glAttachShader(program, vertex)
gl.glAttachShader(program, fragment)
gl.glLinkProgram(program)

if not gl.glGetProgramiv(program, gl.GL_LINK_STATUS):
    print(gl.glGetProgramInfoLog(program))
    raise RuntimeError('Linking error')

gl.glDetachShader(program, vertex)
gl.glDetachShader(program, fragment)

gl.glUseProgram(program)
print("here")
# Build data
data = np.zeros((4,2), dtype=np.float32)

buffer = gl.glGenBuffers(1)

# Make this buffer the default one
gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer)

stride = data.strides[0]

offset = ctypes.c_void_p(0)
loc = gl.glGetAttribLocation(program, "position")
gl.glEnableVertexAttribArray(loc)
gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer)
gl.glVertexAttribPointer(loc, 2, gl.GL_FLOAT, False, stride, offset)

# Assign CPU data
data[...] = (-1,+1), (+1,+1), (-1,-1), (+1,-1)

# Upload CPU data to GPU buffer
gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_DYNAMIC_DRAW)

print("here")
"""