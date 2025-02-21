FROM golang:1.20 AS builder

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .

WORKDIR /app/cmd
RUN go build -o registration_service .

FROM gcr.io/distroless/base

COPY --from=builder /app/cmd/registration_service /registration_service

ENTRYPOINT ["/registration_service"]