package main

import (
    "log"
    "net/http"
    "github.com/gorilla/mux"
    "dog-play/internal/handlers"
)

func main() {
    r := mux.NewRouter()
    
    r.HandleFunc("/play", handlers.PlayHandler).Methods("POST")
    
    http.Handle("/", r)
    
    log.Println("Starting dog-play service on port 8080...")
    if err := http.ListenAndServe(":8080", nil); err != nil {
        log.Fatalf("Could not start server: %s\n", err)
    }
}