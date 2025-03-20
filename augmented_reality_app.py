import cv2
import numpy as np
import sys
import random
import math
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from PIL import Image

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
AR_OBJECT_SIZE = 0.5

# AR Object's vertices
vertices = [
    [ AR_OBJECT_SIZE, AR_OBJECT_SIZE, AR_OBJECT_SIZE ],
    [ -AR_OBJECT_SIZE, AR_OBJECT_SIZE, AR_OBJECT_SIZE ],
    [ -AR_OBJECT_SIZE, -AR_OBJECT_SIZE, AR_OBJECT_SIZE ],
    [ AR_OBJECT_SIZE, -AR_OBJECT_SIZE, AR_OBJECT_SIZE ],
    [ AR_OBJECT_SIZE, AR_OBJECT_SIZE, -AR_OBJECT_SIZE ],
    [ -AR_OBJECT_SIZE, AR_OBJECT_SIZE, -AR_OBJECT_SIZE ],
    [ -AR_OBJECT_SIZE, -AR_OBJECT_SIZE, -AR_OBJECT_SIZE ],
    [ AR_OBJECT_SIZE, -AR_OBJECT_SIZE, -AR_OBJECT_SIZE ],
]

edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)
]

colors = [
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
]

# Camera parameters
camera_pos = [0, 0, -5]
camera_rot = [0, 0, 0]

# Initialize OpenCV and setup camera
cap = cv2.VideoCapture(0)

def draw_object():
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glColor3fv(colors[random.randint(0, len(colors)-1)])
            glVertex3fv(vertices[vertex])
    glEnd()

def update_camera():
    glLoadIdentity()
    gluLookAt(camera_pos[0], camera_pos[1], camera_pos[2],
              camera_pos[0] + math.sin(math.radians(camera_rot[1])),
              camera_pos[1] + math.sin(math.radians(camera_rot[0])),
              camera_pos[2] + math.cos(math.radians(camera_rot[1])),
              0, 1, 0)

def draw_scene():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    update_camera()
    draw_object()
    glutSwapBuffers()

def init_opengl():
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.1, 0.1, 0.1, 1)
    glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, float(WINDOW_WIDTH) / float(WINDOW_HEIGHT), 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)

def process_frame():
    ret, frame = cap.read()
    if not ret:
        print("Error capturing video frame")
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def main_loop():
    while True:
        frame = process_frame()
        if frame is None:
            break
        cv2.imshow('AR View', frame)
        draw_scene()
        
        key = cv2.waitKey(1)
        if key == 27:  # Esc key to exit
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT)
    glutCreateWindow(b'Augmented Reality App')
    init_opengl()
    glutDisplayFunc(draw_scene)
    glutIdleFunc(main_loop)
    main_loop()

if __name__ == '__main__':
    main()