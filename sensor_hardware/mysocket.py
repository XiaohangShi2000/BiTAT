import socket

class MySocket:
    '''demonstration class only
      - coded for clarity, not efficiency
    '''

    def __init__(self, sock=None):
        self.MSGLEN = 192*192
        if sock is None:
            self.sock = socket.socket(
                socket.AF_INET, socket.SOCK_STREAM)
        else:
            self.sock = sock

    def connect(self, host, port):
        self.sock.connect((host, port))

    # def mysend(self, msg):
    #     totalsent = 0
    #     while totalsent < self.MSGLEN:
    #         sent = self.sock.send(msg[totalsent:])
    #         if sent == 0:
    #             raise RuntimeError("socket connection broken")
    #         totalsent = totalsent + sent

    def mysend(self, msg):
        totalsent = 0
        while totalsent < self.MSGLEN:
            sent = self.sock.send(msg[totalsent:])
            if sent == 0:
                raise RuntimeError("socket connection broken")
            totalsent = totalsent + sent

    def myreceive(self, msglen):
        chunks = b''
        bytes_recd = 0
        while bytes_recd < msglen:
            chunk = self.sock.recv(min(msglen - bytes_recd, 2048))
            if not chunk:
                # raise RuntimeError("socket connection broken")
                continue
            else:
                chunks += chunk
                bytes_recd = bytes_recd + len(chunk)
        return chunks
    
    def myreceive_one(self):
        return self.sock.recv(1)
    
    def close(self):
        self.sock.close()