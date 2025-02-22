FROM golang:1.20 AS builder

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .

WORKDIR /app/cmd
RUN go build -o play main.go

FROM gcr.io/distroless/base

COPY --from=builder /app/cmd/play /play

ENTRYPOINT ["/play"]