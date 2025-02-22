package handlers

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"

	"github.com/gorilla/mux"
)

type Dog struct {
	ID    string `json:"id"`
	Name  string `json:"name"`
	Breed string `json:"breed"`
}

// PlayHandler handles play requests for registered dogs.
func PlayHandler(w http.ResponseWriter, r *http.Request) {
	// Logic to handle play action for registered dogs
	// This should check if the dog is registered and then proceed
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("Dog is playing!"))
}

func CanPlay(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	dogID := vars["id"]

	// Check if the dog is registered
	resp, err := http.Get(fmt.Sprintf("http://dog-registration:8084/dogs/%s", dogID))
	if err != nil || resp.StatusCode != http.StatusOK {
		http.Error(w, "This dog cannot play, it should be registered first", http.StatusForbidden)
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
	json.NewEncoder(w).Encode(map[string]string{"message": "This dog can play"})
}

// RegisterRoutes registers the play routes with the provided router.
func RegisterRoutes(router *mux.Router) {
	router.HandleFunc("/play", PlayHandler).Methods("POST")
}

func SetupRoutes(r *mux.Router) {
	r.HandleFunc("/play/{id}", CanPlay).Methods("GET")
}
