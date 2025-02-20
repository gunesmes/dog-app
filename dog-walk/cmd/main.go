package main

import (
	"dog-walk/internal/handlers"
	"log"
	"net/http"
)

func main() {
	http.HandleFunc("/walk", handlers.NewWalkHandler().HandleWalk)

	log.Println("Starting dog-walk service on port 8082...")
	if err := http.ListenAndServe(":8082", nil); err != nil {
		log.Fatalf("Could not start server: %s\n", err)
	}
}
