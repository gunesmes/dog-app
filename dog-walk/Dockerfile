FROM golang:1.20 AS builder

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .

WORKDIR /app/dog-walk/cmd
RUN go build -o dog-walk-service

FROM gcr.io/distroless/base

COPY --from=builder /app/dog-walk/cmd/dog-walk-service /usr/local/bin/dog-walk-service

ENTRYPOINT ["/usr/local/bin/dog-walk-service"]