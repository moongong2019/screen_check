"""
Usage:
# To test whether the andriod device is ready
adb shell LD_LIBRARY_PATH=/data/local/tmp /data/local/tmp/minicap -P 1080x2160@540x1080/90 -t

# Run minicap on andrioid devices
adb shell LD_LIBRARY_PATH=/data/local/tmp /data/local/tmp/minicap -P 1080x2160@540x1080/90

# local forward minicap socket to port #1313
adb forward tcp:1313 localabstract:minicap
"""
import os
import socket
import threading
import sys
import time
import signal
import numpy as np
import cv2

TIME_GAP = 0.1

signal_int_set = False
def Handler(sig_num, frame):
    print("Exit!!!")
    signal_int_set = True
    sys.exit(sig_num)
signal.signal(signal.SIGINT, Handler)


class Get_Image(object):
    def __init__(self, device_no, port, display_shape=None, image_path='image/'):
        self.device_no = device_no
        self.port = port
        if display_shape is not None:
            display_shape = (int(display_shape[0]), int(display_shape[1]))
            cmd = 'adb -s %s shell LD_LIBRARY_PATH=/data/local/tmp /data/local/tmp/minicap -P %dx%d@270x480/90' \
                     % (self.device_no, display_shape[0], display_shape[1])
            print(cmd)
            os.popen(cmd)
            time.sleep(0.5)
            #os.system('adb forward tcp:%d localabstract:minicap' % port)
            os.system('adb -s %s forward tcp:%d localabstract:minicap' % (self.device_no, port))
        else:
            print('display shape should not none...')
        time.sleep(0.5)
        self.connection = socket.create_connection(('127.0.0.1', port))
        self.frame_buffer = []
        self.buffer_lock = threading.Lock()
        self.buffer_lock.acquire()
        self.read_frames_thread=threading.Thread(target=self.Read_Frames_Thread,
                    args=(self.connection, self.frame_buffer, self.buffer_lock))
        self.read_frames_thread.setDaemon(True)
        self.read_frames_thread.start()
        self.image_path = image_path

        def get_jpg_number(path):
            count = 0
            for file in os.listdir(path):
                if '.jpg' in file:
                    count = count+1
            return count

        #self.image_num = get_jpg_number(self.image_path)
        self.image_num = 0

    def Read_Bytes(self, socket, length):
        out = socket.recv(length)
        length -= len(out)
        while length > 0:
            more = socket.recv(length)
            out += more
            length -= len(more)
        return bytearray(out)

    def Read_Frames_Thread(self, socket, frame_buffer, buffer_lock, buffer_max_size=10):
        version = self.Read_Bytes(socket, 1)[0]
        #print("Version {}".format(version))
        banner_length = self.Read_Bytes(socket, 1)[0]
        banner_rest = self.Read_Bytes(socket, banner_length - 2)
        #print("Banner length {}".format(banner_length))

        while True:
            # If receive SIGINT, exit!
            if signal_int_set:
                return

            frame_bytes = list(self.Read_Bytes(socket, 4))
            total = 0
            frame_bytes.reverse()
            for byte in frame_bytes:
                total = (total << 8) + byte
            #print("JPEG data: {}".format(total))
            jpeg_data = self.Read_Bytes(socket, total)

            if len(self.frame_buffer) != 0:
                buffer_lock.acquire()
            frame_buffer.append(bytearray(jpeg_data))
            if len(frame_buffer) > buffer_max_size:
                del(frame_buffer[0])
            buffer_lock.release()

    def Get_Frame(self):
        # Get Image
        self.buffer_lock.acquire()
        image = self.frame_buffer[-1]
        self.buffer_lock.release()
        # Write Image
        self.image_num += 1
        new_image_name = int(time.time()*1000)
        image_out_file = open(self.image_path+str(new_image_name)+'.jpg', 'wb')
        image_out_file.write(bytearray(image))
        # Show Image
        image = cv2.imdecode(np.array(bytearray(image)), 1)
        cv2.imshow('capture', image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            exit(0)

    def Get_Frame_for_Agent(self):
        # Get Image
        self.buffer_lock.acquire()
        image = self.frame_buffer[-1]
        self.buffer_lock.release()
        # Write Image
        self.image_num += 1
        # Show Image
        image = cv2.imdecode(np.array(bytearray(image)), 1)
        return image

    def Get_Frame_for_Agent_undecode(self):
        # Get Image
        self.buffer_lock.acquire()
        image = self.frame_buffer[-1]
        self.buffer_lock.release()
        return image

    def close(self):
        self.connection.close()

def main():

    Device_Input = Get_Image(device_no='', port=1313)


    while True:
        last_time = time.time()

        image = Device_Input.Get_Frame()

        current_time = time.time()
        sleep_time = TIME_GAP - (current_time-last_time)
        if sleep_time > .0:
            time.sleep(sleep_time)


if __name__ == '__main__':
    main()
