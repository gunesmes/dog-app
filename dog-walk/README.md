# Dog Walk Service

The Dog Walk service is part of the Dog Mobile App microservices architecture. This service is responsible for handling requests related to dog walking activities.

## Overview

This service allows registered dogs to be walked. It ensures that only dogs that have been registered can access the walking functionalities.

## Endpoints

- **POST /walk**: Initiates a walk for a registered dog.
- **GET /walk/status**: Retrieves the current status of a dog's walk.

## Dependencies

This service relies on the following packages:
- Gorilla Mux for routing
- Any other necessary packages as defined in `go.mod`

## Running the Service

To run the Dog Walk service, use the following command:

```bash
go run cmd/main.go
```

## Docker

To build and run the Dog Walk service in a Docker container, use the following commands:

```bash
docker build -t dog-walk .
docker run -p 8080:8080 dog-walk
```

## Testing

Make sure to test the service endpoints using tools like Postman or curl to ensure they are functioning as expected.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.