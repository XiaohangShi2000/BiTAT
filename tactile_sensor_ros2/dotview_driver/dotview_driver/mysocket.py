import socket

class MySocket:
    def __init__(self, sock=None, msg_len=192*192+1):
        self.MSGLEN = msg_len
        if sock is None:
            self.sock = socket.socket(
                socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        else:
            self.sock = sock

    def bind(self, host, port):
        self.sock.bind((host, port))
    
    def accept(self):
        self.sock.listen(5)
        conn, addr = self.sock.accept()
        self.conn = conn
        self.addr = addr

    def mysend(self, msg):
        totalsent = 0
        while totalsent < self.MSGLEN:
            sent = self.conn.send(msg[totalsent:])
            if sent == 0:
                raise RuntimeError("socket connection broken")
            totalsent = totalsent + sent
    
    def mysend_one(self, msg):
        sent = self.conn.sendall(msg)
        if sent == 0:
            print("sent one failed")

    def myreceive(self):
        # chunks = []
        chunks = b''
        bytes_recd = 0
        while bytes_recd < self.MSGLEN:
            chunk = self.conn.recv(min(self.MSGLEN - bytes_recd, 8192))
            if chunk == b'':
            # if not chunk:
                raise RuntimeError("socket connection broken")
                # pass
            chunks += chunk
            bytes_recd = bytes_recd + len(chunk)
        return chunks
    
    def close(self):
        self.conn.close() 
        self.sock.close()