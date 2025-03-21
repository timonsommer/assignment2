import moderngl
import moderngl_window as mglw
from pyrr import Matrix44

import cv2
import numpy as np
import os

from prediction import predict, get_camera_matrix, draw_landmarks_on_image, get_fov_y, reproject, solvepnp


class CameraAR(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "CameraAR"
    resource_dir = os.path.normpath(os.path.join(__file__, '../data'))
    previousTime = 0
    currentTime = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    

        # Shader for rendering 3D objects
        self.prog3d = self.ctx.program(
            vertex_shader='''
                #version 330

                uniform mat4 Mvp;

                in vec3 in_position;
                in vec3 in_normal;
                in vec2 in_texcoord_0;

                out vec3 v_vert;
                out vec3 v_norm;
                out vec2 v_text;

                void main() {
                    gl_Position = Mvp * vec4(in_position, 1.0);
                    v_vert = in_position;
                    v_norm = in_normal;
                    v_text = in_texcoord_0;
                }
            ''',
            fragment_shader='''
                #version 330

                uniform vec3 Color;
                uniform vec3 Light;
                uniform sampler2D Texture;
                uniform bool withTexture;

                in vec3 v_vert;
                in vec3 v_norm;
                in vec2 v_text;

                out vec4 f_color;

                void main() {
                    float lum = clamp(dot(normalize(Light - v_vert), normalize(v_norm)), 0.0, 1.0) * 0.8 + 0.2;
                    if (withTexture) {
                        f_color = vec4(Color * texture(Texture, v_text).rgb * lum, 1.0);
                    } else {
                        f_color = vec4(Color * lum, 1.0);
                    }
                }
            ''',
        )
        
        self.rect_shader = self.ctx.program(
            vertex_shader='''
                #version 330

                uniform mat4 Mvp;

                in vec3 in_position;
                in vec2 in_texcoord_0;

                out vec3 v_vert;
                out vec2 v_text;

                void main() {
                    gl_Position = vec4(in_position, 1.0);
                    v_vert = in_position;
                    v_text = in_texcoord_0;
                }
            ''',
            fragment_shader='''
                #version 330

                uniform sampler2D Texture;

                in vec3 v_vert;
                in vec2 v_text;

                out vec4 f_color;

                void main() {
                        f_color = vec4(texture(Texture, v_text).rgb, 1.0);
                }
            ''',
        )
        
        self.mvp = self.prog3d['Mvp']
        self.light = self.prog3d['Light']
        self.color = self.prog3d['Color']
        self.withTexture = self.prog3d['withTexture']

        # Load the 3D virtual object, and the marker for hand landmarks
        self.scene_cube = self.load_scene('crate.obj')
        self.scene_marker = self.load_scene('marker.obj')

        # Extract the VAOs from the scene
        self.vao_cube = self.scene_cube.root_nodes[0].mesh.vao.instance(self.prog3d)
        self.vao_marker = self.scene_marker.root_nodes[0].mesh.vao.instance(self.prog3d)

        # Texture of the cube
        self.texture = self.load_texture_2d('crate.png')

        # Define the initial position of the virtual object
        # The OpenGL camera is position at the origin, and look at the negative Z axis. The object is at 30 centimeters in front of the camera.
        self.object_pos = np.array([0.0, 0.0, -30.0])

        """
        --------------------------------------------------------------------
        TODO: Task 3. 
        Add support to render a rectangle of window size. 
        --------------------------------------------------------------------
        """

        # Start OpenCV camera
        self.capture = cv2.VideoCapture(0)

        # Get a frame to set the window size and aspect ratio
        ret, frame = self.capture.read()
        self.frame_width = frame.shape[1]
        self.frame_height = frame.shape[0]
        self.aspect_ratio = self.frame_width / self.frame_height
        self.camera_mat = get_camera_matrix(self.frame_width, self.frame_height)
        self.fov = get_fov_y(self.camera_mat, self.frame_height)
        self.window_size = (self.frame_width, self.frame_height)
        self.wnd.size = self.window_size
        
        self.texture1 = self.ctx.texture((self.frame_width, self.frame_height), 3)
        
        self.rect = self.ctx.buffer(np.array([
            -1.0, -1.0, 0.0, 0.0,
            1.0, -1.0, 1.0, 0.0,
            -1.0,  1.0, 0.0, 1.0,
            1.0,  1.0, 1.0, 1.0,
        ], dtype='f4'))

        self.rect_indices = self.ctx.buffer(np.array([
            0, 1, 2,
            1, 3, 2,
        ], dtype='i4'))

        self.vao_rect = self.ctx.vertex_array(
            self.rect_shader,
            [(self.rect, '2f 2f', 'in_position', 'in_texcoord_0')],
            self.rect_indices,
        )

    def on_render(self, time: float, frame_time: float):
        self.ctx.clear(1.0, 1.0, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        """
        ---------------------------------------------------------------
        TODO: Task 3. 
        Get OpenCV video frame, display in OpenGL. 
        Render the frame to a screen-sized rectange. 
        ---------------------------------------------------------------
        """
        
        fps = 1 / frame_time

        ret, frame = self.capture.read()
        # Solve the landmarks in world space
        detection_result = predict(frame)
        hand_landmarks_list = detection_result.multi_hand_landmarks
        hand_world_landmarks_list = detection_result.multi_hand_world_landmarks
        world_landmarks_list = solvepnp(hand_world_landmarks_list, hand_landmarks_list,
                                                self.camera_mat, self.frame_width, self.frame_height)
        if hand_landmarks_list is not None or hand_world_landmarks_list is not None:
            reprojection_error, reprojection_points_list = reproject(world_landmarks_list, hand_landmarks_list, self.camera_mat, self.frame_width, self.frame_height)
            if reprojection_error < 10.0:
                for hand_landmarks in reprojection_points_list:
                    for l in hand_landmarks:
                        cv2.circle(frame, (int(l[0]), int(l[1])), 3, (0, 0, 255), 2)
                        
        cv2.putText(frame, str(int(fps))+" FPS", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frame = cv2.flip(frame, 0)
        frame = cv2.flip(frame, 1)
        self.texture1.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes())
        self.texture1.use()

        # make rect fill the screen
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.vao_rect.render()
        self.ctx.enable(moderngl.DEPTH_TEST)

        """
        ---------------------------------------------------------------
        TODO: Task 4.
        Perform hand landmark prediction, and 
        solve PnP to get world landmarks list.
        ---------------------------------------------------------------
        """
        
        # world_landmarks_list *= -100
        for world_landmarks in world_landmarks_list:
            for landmark in world_landmarks:
                landmark[0] *= -100
                landmark[1] *= -100
                landmark[2] *= -100
        
        # OpenCV to OpenGL conversion
        # The world points from OpenCV need some changes to be OpenGL ready.
        # First, the model points are in meters (MediaPipe convention), while our camera matrix is in units. There exists a scale ambiguity of the true hand landmarks, i.e., if we scale up the world points by 1000, its projection remains the same (due to perspective division).
        # Here we shift the measurement from meter to centimeter, and assume our world space in OpenGL is in centimeters, just for easy visualization and object interaction. So we multiply all points by 100.

        # Second, the OpenCV and OpenGL camera coordinate system are different. # OpenCV: right x, down y, into screen z. Image: right x, down y.
        # OpenGL: right x, up y, out of screen z. Image: right x, up y.
        # Check for image and 3D points flip to make sure the points are properly converted.

        """
        ----------------------------------------------------------------------
        TODO: Task 5.
        We detect a simple pinch gesture, and check if the index finger hits 
        the cube. We approximate by just checking the finger tip is close 
        enough to the cube location.
        ----------------------------------------------------------------------
        """
        pinched = False
        grabbed = False
        indexPos = [0,0,0]
        
        pinch_threshold = 3.0
        grab_threshold = 8.0
        for world_landmarks in world_landmarks_list:
            if not pinched and np.linalg.norm(world_landmarks[4] - world_landmarks[8]) <= pinch_threshold:
                pinched = True
            if not grabbed and np.linalg.norm(world_landmarks[8] - self.object_pos) <= grab_threshold:
                grabbed = True
                indexPos = world_landmarks[8]

        """
        ----------------------------------------------------------------------
        TODO: Task 4. 
        Render the markers.
        ----------------------------------------------------------------------
        """
        # Note we have to set the OpenGL projection matrix by following parameters from the OpenCV camera matrix, i.e., the field of view.
        # You can use Matrix44.perspective_projection function, and set the parameters accordingly. Note that the fov must be computed based on the camera matrix. See prediction.py.

        # In this example, a random FOV value is set. Do not use this value in your final program.
        proj = Matrix44.perspective_projection(self.fov, self.aspect_ratio, 0.1, 1000)

        # Translate the object to its position
        translate = Matrix44.from_translation(self.object_pos)

        # Add a bit of random rotation just to be dynamic
        rotate = Matrix44.from_y_rotation(np.sin(time) * 0.5 + 0.2)

        # Scale the object up for easy viewing

        self.color.value = (1.0, 1.0, 1.0)
        scale = Matrix44.from_scale((3, 3, 3))
        if pinched:  # A bit of feedback when the object is grabbed
            self.color.value = (1.0, 0.0, 0.0)
            if grabbed:
                self.object_pos = indexPos
                scale = Matrix44.from_scale((2, 2, 2))
        mvp = proj * translate * rotate * scale
        self.light.value = (10, 10, 10)
        self.mvp.write(mvp.astype('f4'))
        self.withTexture.value = True

        # Render the object
        self.texture.use()
        self.vao_cube.render()

        # Render the landmarks
        self.color.value = (0.0, 1.0, 0.0)
        self.withTexture.value = False
        for world_landmarks in world_landmarks_list:
            for landmark in world_landmarks:
                translate = Matrix44.from_translation(landmark)
                scale = Matrix44.from_scale((0.3, 0.3, 0.3))
                mvp = proj * translate * scale
                self.mvp.write(mvp.astype('f4'))
                self.vao_marker.render()
                
        del world_landmarks_list
        del detection_result
        


if __name__ == '__main__':
    CameraAR.run()
