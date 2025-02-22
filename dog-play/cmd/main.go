package main

import (
	"dog-play/internal/handlers"
	"github.com/gorilla/mux"
	"log"
	"net/http"
)

func main() {
	r := mux.NewRouter()
	handlers.SetupRoutes(r)
	log.Println("Starting dog-play service on port 8082...")
	log.Fatal(http.ListenAndServe(":8082", r))
}
