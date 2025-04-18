package main

import (
	"dog-registration/internal/handlers"
	"log"
	"net/http"

	"github.com/gorilla/mux"
)

func main() {
	r := mux.NewRouter()
	r.HandleFunc("/register", handlers.RegisterDog).Methods("POST")
	r.HandleFunc("/dogs", handlers.GetRegisteredDogs).Methods("GET")
	r.HandleFunc("/dogs/{dogID}", handlers.GetDogByID).Methods("GET")
	r.HandleFunc("/dogs", handlers.DeleteAllDogs).Methods("DELETE")

	log.Println("Starting dog-registration service on :8084")
	if err := http.ListenAndServe(":8084", r); err != nil {
		log.Fatalf("Could not start server: %s\n", err)
	}
}
