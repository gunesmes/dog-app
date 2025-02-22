package handlers

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"

	"github.com/gorilla/mux"
)

// WalkHandler handles walk-related requests
type WalkHandler struct{}

// NewWalkHandler creates a new WalkHandler
func NewWalkHandler() *WalkHandler {
	return &WalkHandler{}
}

// HandleWalk handles the walk action for registered dogs
func (h *WalkHandler) HandleWalk(w http.ResponseWriter, r *http.Request) {
	// Logic to handle the walk request for registered dogs
	// This should check if the dog is registered and then proceed
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("Dog is walking!"))
}

// RegisterRoutes registers the walk routes
func (h *WalkHandler) RegisterRoutes(r *mux.Router) {
	r.HandleFunc("/walk", h.HandleWalk).Methods("POST")
}

type Dog struct {
	ID    string `json:"id"`
	Name  string `json:"name"`
	Breed string `json:"breed"`
}

func CanWalk(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	dogID := vars["id"]

	// Check if the dog is registered
	resp, err := http.Get(fmt.Sprintf("http://dog-registration:8084/dogs/%s", dogID))
	if err != nil || resp.StatusCode != http.StatusOK {
		http.Error(w, "This dog cannot walk, it should be registered first", http.StatusForbidden)
		return
	}

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		http.Error(w, "Error reading response from registration service", http.StatusInternalServerError)
		return
	}

	var dog Dog
	err = json.Unmarshal(body, &dog)
	if err != nil {
		http.Error(w, "Error parsing response from registration service", http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{"message": "This dog can walk"})
}

func SetupRoutes(r *mux.Router) {
	r.HandleFunc("/walk/{id}", CanWalk).Methods("GET")
}
