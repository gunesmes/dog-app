# Dog Sleep Service

This is the documentation for the **Dog Sleep** microservice, which is part of the Dog Mobile App architecture. The Dog Sleep service is responsible for managing sleep-related actions for registered dogs.

## Overview

The Dog Sleep service allows registered dogs to log their sleep activities. It provides endpoints to handle sleep requests and interacts with the dog registration service to ensure that only registered dogs can access its functionalities.

## Features

- Log sleep activities for registered dogs.
- Retrieve sleep data for specific dogs.
- Ensure that only registered dogs can log sleep activities.

## Endpoints

- `POST /sleep` - Log sleep activity for a registered dog.
- `GET /sleep/{dogId}` - Retrieve sleep data for a specific dog.

## Getting Started

### Prerequisites

- Go 1.16 or later
- Docker (for containerization)

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd dog-mobile-app/dog-sleep
   ```

2. Install dependencies:
   ```
   go mod tidy
   ```

### Running the Service

To run the Dog Sleep service locally, use the following command:

```
go run cmd/main.go
```

### Docker

To build and run the Dog Sleep service in a Docker container, use the following commands:

1. Build the Docker image:
   ```
   docker build -t dog-sleep .
   ```

2. Run the Docker container:
   ```
   docker run -p 8080:8080 dog-sleep
   ```

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.