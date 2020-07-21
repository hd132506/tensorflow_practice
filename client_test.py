import socket
import numpy as np

filename = 'result.csv'
result = np.zeros(30)

for epoch in range(30):
    result[epoch] = 1.

np.savetxt(filename, result)

ServerIp = '166.104.245.218'

# Now we can create socket object
s = socket.socket()

# Lets choose one port and connect to that port
PORT = 11821

# Lets connect to that port where server may be running
s.connect((ServerIp, PORT))

# We can send file sample.txt
file = open("result.csv", "rb")
SendData = file.read(1024)


while SendData:
    #Now send the content of sample.txt to server
    s.send(SendData)
    SendData = file.read(1024)      

# Close the connection from client side
s.close()
file.close()
