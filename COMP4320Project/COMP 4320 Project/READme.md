Joshua Kim
COMP 4320 Project - Dr. Li
July 30, 2023

My client.c and server.c files are the only files necessary for compilation. I developed this project using ubuntu through VirtualBox and my gcc compiler is version 11.3.0. To compile, run the command "gcc server.c -o server" and "gcc client.c -o client". Start up the server first by running 
./server and then run ./client. Everytime want to re-compile the programs, you must change the port number to a unique number or you will run into an error binding issue because each socket is binded to a port. This will generated the receivedFile.txt, indicating that the packets have been received by the client. These are the following resources that I referenced:

https://www.scaler.com/topics/udp-server-client-implementation-in-c/
https://idiotdeveloper.com/udp-client-server-implementation-in-c/

Also included in my submission are clientOutput.txt and serverOutput.txt, which are my scripts of the execution traces, a python script to generate a 90 kB ASCII file, and TextFile, which is the ASCII file and file sent by the server.