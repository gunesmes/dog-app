package main

import (
	"dog-registration/internal/handlers"
	"log"
	"net/http"
)

func main() {
	http.HandleFunc("/register", handlers.RegisterDog)
	http.HandleFunc("/dogs", handlers.GetRegisteredDogs)

	log.Println("Starting dog-registration service on :8080")
	if err := http.ListenAndServe(":8080", nil); err != nil {
		log.Fatalf("Could not start server: %s\n", err)
	}
}
