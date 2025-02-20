FROM golang:1.20 AS builder

WORKDIR /app

COPY go.mod ./
COPY go.sum ./
RUN go mod download

COPY . .

RUN go build -o play cmd/main.go

FROM gcr.io/distroless/base

COPY --from=builder /app/play /app/play

CMD ["/app/play"]