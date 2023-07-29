#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>

#define PORT 5023
#define BUFFER_SIZE 512
#define FILENAME "TextFile"

void segmentation(int sockfd) {
    // Initializes buffer size, IP information, and size of client_addr
    char buffer[BUFFER_SIZE];
    struct sockaddr_in client_addr;
    socklen_t addr_len = sizeof(client_addr);

    // Receive FTP request from the client
    memset(buffer, 0, sizeof(buffer));
    int bytes_received = recvfrom(sockfd, buffer, BUFFER_SIZE, 0, (struct sockaddr*)&client_addr, &addr_len);

    if (bytes_received < 0) {
        perror("Error receiving data");
        close(sockfd);
        exit(EXIT_FAILURE);
    }

    printf("FTP Request received from %s:%d\n", inet_ntoa(client_addr.sin_addr), ntohs(client_addr.sin_port));
    printf("Data received: %s\n", buffer);

    // Check if the client sent a GET request
    if (strcmp(buffer, "GET TextFile") == 0) {
        FILE* file = fopen(FILENAME, "rb");

        printf("Reading the file '%s'...\n", FILENAME);

        // Read the file content into the buffer and send it to the client
        size_t bytesRead;
        int sequenceNumber = 0;

        // Loop reads file contents into buffer in chunks of 512 bytes
        while ((bytesRead = fread(buffer + 1, 1, BUFFER_SIZE - 1, file)) > 0) {
            // Set the first byte as the sequence number
            buffer[0] = (char)sequenceNumber;

            // If the remaining data is less than 512 bytes, pad the last packet with NULL characters
            if (bytesRead < BUFFER_SIZE - 1) {
                memset(buffer + bytesRead + 1, '\0', BUFFER_SIZE - bytesRead - 1);
            }

            sendto(sockfd, buffer, BUFFER_SIZE, 0, (struct sockaddr*)&client_addr, sizeof(client_addr));
            printf("Sent packet with sequence number %d and data: %.64s\n", sequenceNumber, buffer + 1);
            sequenceNumber++;
        }

        printf("File transfer complete.\n");

        // Send the end-of-file indicator
        buffer[0] = (char)-1;
        sendto(sockfd, buffer, 1, 0, (struct sockaddr*)&client_addr, sizeof(client_addr));
        printf("Reached end of file.\n");

        fclose(file);
    }
}

int main() {
    int sockfd;
    struct sockaddr_in server_addr;
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);

    // Initialize server address and port
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(PORT);

    // Bind the socket to the specified port
    if (bind(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("Error binding");
        close(sockfd);
        exit(EXIT_FAILURE);
    }

    printf("FTP Server started and listening on port %d...\n", PORT);

    while (1) {
        segmentation(sockfd);
    }

    close(sockfd);
    return 0;
}