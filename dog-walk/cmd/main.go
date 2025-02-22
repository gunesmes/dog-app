package main

import (
	"dog-walk/internal/handlers"
	"log"
	"net/http"

	"github.com/gorilla/mux"
)

func main() {
	r := mux.NewRouter()
	handlers.SetupRoutes(r)
	log.Println("Starting dog-walk service on port :8083")
	log.Fatal(http.ListenAndServe(":8083", r))
}
