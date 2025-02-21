package handlers

import (
	"github.com/gorilla/mux"
	"net/http"
)

// PlayHandler handles play requests for registered dogs.
func PlayHandler(w http.ResponseWriter, r *http.Request) {
	// Logic to handle play action for registered dogs
	// This should check if the dog is registered and then proceed
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("Dog is playing!"))
}

// RegisterRoutes registers the play routes with the provided router.
func RegisterRoutes(router *mux.Router) {
	router.HandleFunc("/play", PlayHandler).Methods("POST")
}
