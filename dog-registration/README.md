# Dog Registration Service

This service is responsible for registering dogs and managing their data. It ensures that only registered dogs can access the play and walk services.

## Features

- Register new dogs
- Retrieve registered dog information
- Validate dog registration for play and walk services

## Endpoints

- `POST /register`: Register a new dog
- `GET /dogs/{id}`: Retrieve information about a registered dog

## Getting Started

To run the dog-registration service locally, follow these steps:

1. Clone the repository:
   ```
   git clone <repository-url>
   cd dog-mobile-app/dog-registration
   ```

2. Build the service:
   ```
   go build -o registration cmd/main.go
   ```

3. Run the service:
   ```
   ./registration
   ```

## Docker

To build and run the service using Docker, use the following commands:

1. Build the Docker image:
   ```
   docker build -t dog-registration .
   ```

2. Run the Docker container:
   ```
   docker run -p 8080:8080 dog-registration
   ```

## Dependencies

This service uses the following dependencies (defined in `go.mod`):

- [Gin](https://github.com/gin-gonic/gin): A web framework for Go
- [Gorm](https://gorm.io/index.html): An ORM for Go

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.