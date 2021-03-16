import numpy as np
import body
#import glfw
#import pyrr


#Далее константы, нужные для условно аналитического решения
eps = 0.00001
G  = 1
hi_2 = None # хи в квадрате, характеристика  дифференцильнрго уравнения
h = None # константа энергии
r = None  # радиус вектор r = r2 - r1
r_v = None # производная dr/dt
c = None # двойная секторная скорость
E = None # вектор лапласа
i = None # угол между  вектором с и осью Oz
Ω = None # угол между линией узлов и осью Ох
g = None # угол пороворота Ох до совпадения с перицентром
θ = None
p = None # фокальный параметр
e = None # эксцентриситет орбиты
a = None# Большая полуось
A = None # матрица поворота от системы О к конечной
direction = None
# ввод начальных данных: скорости первого тела, второго тела, начальное положение
b1 = body.Body() # первое тело
b2 = body.Body() # второе тело
bc = body.Body() # центр масс

def print_vector(vn ,v):
    print(vn + " =  [{0},{1},{2}]".format(v[0], v[1], v[2]))

def module(v):
    return np.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])

def compute_angle_from_vectors(v1,v2):
        return np.arccos(np.inner(v1,v2)/(module(v1)*module(v2)))
# инициализируем начальные данные
def init_data_from_file():
    file = open ("data.txt", "r")
    print("m первого тела ")
    b1.m = int(file.readline())
    print("v0 первого тела ")
    b1.v = np.array ([i for i in file.readline().split()], float)
    print("r0 первого тела ")
    b1.r = np.array ([i for i in file.readline().split()], float)
    print("---Перво тело----")
    b1.info()
    print("m второго тела ")
    b2.m = int(file.readline())
    print("v0 второго тела ")
    b2.v = np.array ([i for i in file.readline().split()], float)
    print("r0 второго тела ")
    b2.r = np.array ([i for i in file.readline().split()], float)
    print("---Второе тело----")
    b2.info()
    file.close()
def init_data():
    print("m первого тела ")
    b1.m = float(input())
    print("v0 первого тела ")
    b1.v = np.array(input().split(" "),float)
    print("r0 первого тела ")
    b1.r = np.array(input().split(" "),float)

    print("---Перво тело----")
    b1.info()

    print("m второго тела ")
    b2.m = float(input())
    print("v0 второго  тела ")
    b2.v = np.array(input().split(" "),float)
    print("r0 второго тела ")
    b2.r = np.array(input().split(" "),float)

    print("---Второе  тело----")
    b2.info()

    #Вычисляем константы
    #находим радиус вектор центра масс


def rotate_by_x(ϕ):
    sin = np.sin
    cos = np.cos
    return np.array([[1,0,0],[0,cos(ϕ),-sin(ϕ)],[0, sin(ϕ),cos(ϕ)]],float)
def rotate_by_y(ϕ):
    sin = np.sin
    cos = np.cos
    return np.array([[cos(ϕ),0,sin(ϕ)],[0,1,0],[-sin(ϕ), 0, cos(ϕ)]],float)
def rotate_by_z(ϕ):
    sin = np.sin
    cos = np.cos
    return np.array([[cos(ϕ),-sin(ϕ),0],[sin(ϕ),cos(ϕ),0],[0, 0, 1]],float)
# функция для вычисления всех констант для явного аналитического решения
def compute_data():
    global p, e, g, i, Ω, direction,c, A, hi_2, r, θ
    def compute_general_consts():
        bc.m = (b1.m + b2.m)
        bc.r = (b1.r * b1.m + b2.r * b2.m)/ bc.m

        #скорость центра масс,  центр масс движется прямолинейно и равномерно -  система изолирована
        bc.v = (b1.v * b1.m + b2.v * b2.m)/ bc.m
        bc.info()

        global hi_2, r, r_v, h , c ,E
        # хи в квадрате
        hi_2 = G * bc.m
        # находим радиус вектор r и скорость v_r

        r = b2.r - b1.r
        r_v = b2.v - b1.v
        print_vector("Вектор r:", r)
        print_vector("Вектор cкорости r_v:",r_v)

        c = np.cross(r,r_v)
        h = pow(module(r_v),2)/2 - hi_2/module(r)
        E = np.cross(r_v,c)/hi_2 - r/module(r)
        print_vector("Вектор секторной скорости: ", c)
        print_vector("Вектор Лапласа: ", E)
        print('Консанта интеграла энергии: {}'.format(h))

    compute_general_consts()

    def compute_orbit_oriantaion():
        global i, Ω, direction,c
        i = compute_angle_from_vectors(c, [0.0,0.0,1.0])
        # q_intersect направлящий вектор прямой пересесения  OXY с с1X+c2Y +c3z=0
        q_intersect = np.cross(c, [0.0,0.0,1.0])
        if(i<eps or np.fabs(np.pi - i)<eps):
            print("Линия узлов не определена, Ω = undefinite")
        else:
            Ω = compute_angle_from_vectors(q_intersect, [1.0, 0.0, 0.0])
        if(0<= i and i < np.pi/2):
            print("Движение происходит с запада на восток")
            direction = "WtoE"
        if (np.pi/2 < i and i < np.pi):
            print("Движение происходит с востока на запад")
            direction = "EtoW"
        print("Ω : {}".format(Ω))
        print("i : {}".format(i))

    compute_orbit_oriantaion()

    def determinate_traectory():
        global p,e,g,c,a
        p = pow(module(c),2) / hi_2
        e = np.sqrt(1+ 2*h*pow(module(c),2)/(hi_2*hi_2))
        a = p/(1-e*e)
        print("Фокальный параметр: ",p)
        print("Эксцентриситет: ",e)
        print("Большая полуось: ",a)
        if(0<=e and e<1):
            print("Эллипс")
        elif(np.fabs(1-e)<eps):
            print("Парабола")
        else:
            print("Гипербола")
    determinate_traectory()


    print("Поворачиваем относительно x на угол i:{0}".format(i))
    A = rotate_by_x(i)
    A = np.linalg.inv(A)
    r = np.dot(A, np.transpose(r))
    ##c = np.dot(A, np.transpose(c))
    print()
    print_vector("r ", r)
    u  = np.arccos(r[0]/module(r))#угол в полярной системе коордиант
    w = 1/module(r)  - hi_2/pow(module(c),2)
    A = np.sqrt(hi_2**2/pow(module(c),4) + 2*h/pow(module(c),2))
    θ = np.arccos(w/A)# тэта - аномалия
    g = u  - θ
    print("g:{0}".format(g))

    cos = np.cos
    sin = np.sin

    if(Ω!= None):
        A = np.dot(rotate_by_z(Ω),rotate_by_x(i))
    else:
        print("Поворачиваем плоскость на угол g:{0}".format(g))
        A = rotate_by_z(g)

    A = np.linalg.inv(A)
    r = np.dot(A, np.transpose(r))

    A = np.dot(rotate_by_x(i),rotate_by_z(g)) # финальная матрца преобразования для оставшихся векторов
    A = np.linalg.inv(A)
    c = np.dot(A, np.transpose(c))
    print_vector("c ", c)
    print(r)
    #print(np.dot(A,np.array([[c[0],0,0],[c[1],0,0],[c[2],0,0]])))
    print()





#init()
#compute_general_consts()
#compute_orbit_oriantaion()




"""
if not glfw.init():
    raise Exception("glfw can not be initialized!")

# Configure the OpenGL context.
# If we are planning to use anything above 2.1 we must at least
# request a 3.3 core context to make this work across platforms.
glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
# 4 MSAA is a good default with wide support
glfw.window_hint(glfw.SAMPLES, 4)

# creating the window
window = glfw.create_window(1280, 720, "My OpenGL window", None, None)

# check if window was created
if not window:
    glfw.terminate()
    raise Exception("glfw window can not be created!")

# Query the actual framebuffer size so we can set the right viewport later
# -> glViewport(0, 0, framebuffer_size[0], framebuffer_size[1])
framebuffer_size = glfw.get_framebuffer_size(window)

# set window's position
glfw.set_window_pos(window, 400, 200)

# make the context current
glfw.make_context_current(window)


# the main application loop
while not glfw.window_should_close(window):
    glfw.poll_events()
    glfw.swap_buffers(window)

# terminate glfw, free up allocated resources
glfw.terminate()
"""
