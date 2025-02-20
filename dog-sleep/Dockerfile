FROM golang:1.20 AS builder

WORKDIR /app

COPY go.mod ./
COPY go.sum ./
RUN go mod download

COPY . .

RUN go build -o dog-sleep ./cmd/main.go

FROM gcr.io/distroless/base

COPY --from=builder /app/dog-sleep /usr/local/bin/dog-sleep

CMD ["dog-sleep"]