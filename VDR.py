# import cv2
# from networktables import NetworkTables

# class CameraStream:
#     def __init__(self, camera_index=0, width=1920, height=1080):
#         self.camera_index = camera_index
#         self.width = width
#         self.height = height
#         self.capture = cv2.VideoCapture(camera_index)
#         self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
#         self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
#         self.frame = None

#     def read_frame(self):
#         ret, self.frame = self.capture.read()
#         if not ret:
#             raise RuntimeError("Failed to grab frame.")
#         return self.frame

#     def show_frame(self):
#         cv2.imshow("Camera Feed", self.frame)

#     def check_key(self, key):
#         return cv2.waitKey(1) & 0xFF == ord(key)

#     def main_loop(self):
#         self.read_frame()
#         self.show_frame()

#     def release(self):
#         self.capture.release()

# class NetworkCamera:
#     def __init__(self, url):
#         self.url = url

#     def get_frame(self):
#         pass

# class CaneraProcessor:
#     def __init__(self, camera_stream):
#         self.camera_stream = camera_stream

# class NTables:
#     def __init__(self, test=False, server_ip="10.0.67.2"):
#         if test:
#             NetworkTables.initialize(server="127.0.0.1")
#         else:
#             NetworkTables.initialize(server=server_ip)

#         self.sd = NetworkTables.getTable("SmartDashboard")

#     def get_value(self, key="Enabled", default=None):
#         return self.sd.getValue(key, default)



# if __name__ == "__main__":
#     camera = CameraStream()

#     camera.main_loop()

#     while True:
#         camera.main_loop()

#         if camera.check_key('q'):
#             break
            