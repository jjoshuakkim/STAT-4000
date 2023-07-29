#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>

#define SERVER_IP "127.0.0.1"
#define PORT 5023
#define BUFFER_SIZE 512
#define FILENAME "receivedFile.txt"

// Function is used to reassemble the file from server packets
void reassemble_file(int sockfd) {
    FILE* file = fopen(FILENAME, "wb");

    char buffer[BUFFER_SIZE];
    int sequenceNumber = 0;

    while (1) {
        // Receive the packet from the server
        struct sockaddr_in server_addr;
        socklen_t addr_len = sizeof(server_addr);

        // Returns the number of bytes received in bytesReceived
        ssize_t bytesReceived = recvfrom(sockfd, buffer, BUFFER_SIZE, 0, (struct sockaddr*)&server_addr, &addr_len);
        if (bytesReceived < 0) {
            perror("Error receiving data");
            fclose(file);
            exit(EXIT_FAILURE);
        }

        // Check if it's the end-of-file indicator
        if (bytesReceived == 1 && buffer[0] == (char)-1) {
            printf("End of file received. File transfer complete.\n");
            break;
        }

        // Check the sequence number and write data to receivedFile.txt
        if (buffer[0] == (char)sequenceNumber) {
            fwrite(buffer + 1, 1, bytesReceived - 1, file);
            // Prints the first 64 bytes of the packet
            printf("Received packet with sequence number %d and first 64 bytes of data: %.64s\n", sequenceNumber, buffer + 1);
            // Makes sure that the sequence number is within 0 to 255
            sequenceNumber = (sequenceNumber + 1) % 256;
        } else {
            printf("Out-of-order packet with sequence number %d. Ignoring...\n", buffer[0]);
        }
    }

    fclose(file);
}

int main() {
    int sockfd;
    struct sockaddr_in server_addr;

    // Creates UDP socket
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);

    // Initialize IP and port to communicate with server
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = inet_addr(SERVER_IP);
    server_addr.sin_port = htons(PORT);

    // Send GET request to the server
    const char* request = "GET TextFile";
    sendto(sockfd, request, strlen(request), 0, (struct sockaddr*)&server_addr, sizeof(server_addr));

    // Reassemble the file from received packets
    reassemble_file(sockfd);

    close(sockfd);
    return 0;
}