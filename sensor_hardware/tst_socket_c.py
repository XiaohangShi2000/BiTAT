import socket
import numpy as np
import time
from sensor import FpcSensor
from mysocket import MySocket
import sys

ch = int(sys.argv[1])
ver = int(sys.argv[2])
# s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
# s.connect('./tt.d')
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# host = socket.gethostbyname(socket.gethostname())
# host = '192.168.123.65'
host = '10.42.0.1'
port = 24444 + ch
print("host: ", host)
print("port: ", port)
ts = time.time()
isCon = False
while True:
    try:
        s.connect((host, port))
    except:
        tt = time.time()
        if tt - ts > 5: 
            isCon = False
            break
        else: continue
    else:
        isCon = True
        break

if isCon:
    client_socket = MySocket(sock=s)

    fpc_dev = FpcSensor(ch, ver)
    fpc_dev.sensor_init()
    fpc_dev.get_id()
    fpc_dev.activate_idle()
    fpc_dev.detect_ir()
    fpc_dev.read_ir_clc()

    # data = [0x01] * 192*192
    # to_send = bytes(data)

    # for i in range(5000):
    while True:
        #start = time.time()
        sig = client_socket.myreceive_one()
        if sig == b'\x01':
            # start = time.time()
            ret, pic = fpc_dev.get_image()
            pic = list(pic)
            # end = time.time()
            # print("send cost: ", end-start)
            if ret:
                if len(pic) != fpc_dev.PIXNUM:
                    print("pic length error!")
                    continue
                pic.append(0x01)
            else: 
                pic.append(0x00)
            to_send = bytes(pic)
            client_socket.mysend(to_send)
            
        # end = time.time()
        # print("send cost: ", end-start)
        elif sig == b'\x03':
            client_socket.close()
            fpc_dev.sensor_terminate()
            print("connection ended")
            break
        else:
            pass
else:
    print("Connect to Server timeout!")


