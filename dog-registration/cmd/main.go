package main

import (
	"dog-registration/internal/handlers"
	"log"
	"net/http"
)

func main() {
	http.HandleFunc("/register", handlers.RegisterDog)
	http.HandleFunc("/dogs", handlers.GetRegisteredDogs)

	log.Println("Starting dog-registration service on :8084")
	if err := http.ListenAndServe(":8084", nil); err != nil {
		log.Fatalf("Could not start server: %s\n", err)
	}
}
