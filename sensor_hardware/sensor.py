# from PyQt5.QtCore import QObject, QThread, pyqtSignal
import spidev
import RPi.GPIO as GPIO
import time
import numpy as np

class FpcSensor():
    # new_fpc_data = pyqtSignal(tuple)
    def __init__(self, ch, version):
        # super().__init__()
        
        if   version == 1020: self.__PIXSIZE = 192
        elif version == 1021: self.__PIXSIZE = 160
        else: self.__PIXSIZE = 192
        self.PIXNUM = self.__PIXSIZE**2
        
        self._ch = ch
        if ch == 0:
            self.__IRQ = 25
            self.__CSN = 24
        elif ch == 1:
            self.__IRQ = 18
            self.__CSN = 17
        elif ch == 2:
            self.__IRQ = 22
            self.__CSN = 27
        elif ch == 3:
            self.__IRQ = 26
            self.__CSN = 16
        else: 
            self.__IRQ = 25
            self.__CSN = 24
        self.__SPEED = 12*1000000
        # self.__SPEED = 12000000
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.__CSN, GPIO.OUT, initial=GPIO.HIGH)
        GPIO.setup(self.__IRQ, GPIO.IN, pull_up_down=GPIO.PUD_UP)


    def sensor_init(self):
        spi = spidev.SpiDev()
        if self._ch == 1:   # spi3
            bus = 3
        elif self._ch == 0: # spi0
            bus = 0
        elif self._ch == 2: # spi5
            bus = 5
        elif self._ch == 3: # spi6
            bus = 6
        else: bus = 0
        spi.open(bus, 0)
        spi.mode = 0b00
        spi.max_speed_hz = self.__SPEED
        self._spi = spi
        pass

    def sensor_terminate(self):
        self._spi.close()
        GPIO.cleanup((self.__CSN, self.__IRQ))
        pass

    def get_id(self):
        GPIO.output(self.__CSN, 0)
        rx_data = self._spi.xfer([0xfc, 0x00, 0x00])
        print("ID = 0x%X%X" % (rx_data[1], rx_data[2]))
        GPIO.output(self.__CSN, 1)
    
    def read_ir_clc(self):
        GPIO.output(self.__CSN, 0)
        rx_data = self._spi.xfer([0x1c, 0x00])
        GPIO.output(self.__CSN, 1)
        
    def detect_ir(self):
        epoch = 0
        while True:
            if GPIO.input(self.__IRQ):
                break
            else:
                epoch += 1
                if epoch > 10000:
                    print("Cannot detect interrupt!!")
                    break
    
    def activate_idle(self):
        GPIO.output(self.__CSN, 0)
        rx_data = self._spi.xfer([0x34])
        GPIO.output(self.__CSN, 1)

    def capture_image(self):
        GPIO.output(self.__CSN, 0)
        rx_data = self._spi.xfer([0xc0])
        GPIO.output(self.__CSN, 1)
    
    def get_image(self):
        self.capture_image()
        self.detect_ir()
        self.read_ir_clc()
        GPIO.output(self.__CSN, 0)
        ret = 0
        # rx_data = self._spi.xfer([0xc4, 0x00])
        self._spi.writebytes([0xc4, 0x00])
        to_send = [0x00] * self.PIXNUM
        # to_send = np.zeros(self.PIXNUM, dtype=np.uint8).tolist()
        pic = []
        try:
            # print("do xfer3")
            pic = self._spi.xfer3(to_send)
            # print("xfer3 done")
            ret = 1
        except:
            print("spi connection timed out!")
            pic = tuple(to_send)
            ret = 0
         
        # total_pix = 0
        # pic = []
        # while total_pix < self.PIXNUM:
        #     to_send = [0x00] * min(self.PIXNUM-total_pix, 4096)
        #     chunk = self._spi.xfer2(to_send)
        #     pic += chunk
        #     total_pix += len(chunk)

        # self._pic = self._spi.xfer3(to_send)
        GPIO.output(self.__CSN, 1)
        return ret, pic

    def run(self):
        self.sensor_init()
        self.get_id()
        self.activate_idle()
        self.detect_ir()
        self.read_ir_clc()

        
        count = 0
        while True:
            # s = time.time()
            pic = self.get_image()
            # QThread.msleep(3)
            # count += 1
            # if count == 100:
            #     e = time.time()
            #     print("time: ", e-s)           
            
            # self.new_fpc_data.emit(self._pic)
           #  self.new_fpc_data.emit(pic)
            # e = time.time()
            # print("time: ", e-s)   
