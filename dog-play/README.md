# Dog Play Service

This service handles play-related requests for registered dogs. It ensures that only dogs that have been registered can engage in play activities.

## Features

- **Play Actions**: Allows registered dogs to play.
- **Handler Functions**: Exports functions to manage play requests.

## Getting Started

To run the dog-play service, follow these steps:

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd dog-mobile-app/dog-play
   ```

2. **Build the Docker image**:
   ```
   docker build -t dog-play .
   ```

3. **Run the Docker container**:
   ```
   docker run -p 8080:8080 dog-play
   ```

## API Endpoints

- **POST /play**: Initiates a play session for a registered dog.
- **GET /play/status**: Retrieves the current play status of a registered dog.

## Dependencies

This service relies on the following Go packages:
- [Gin](https://github.com/gin-gonic/gin): A web framework for building APIs in Go.

## License

This project is licensed under the MIT License. See the LICENSE file for details.