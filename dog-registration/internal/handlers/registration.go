package handlers

import (
	"encoding/json"
	"github.com/gorilla/mux"
	"net/http"
)

type Dog struct {
	ID    string `json:"id"`
	Name  string `json:"name"`
	Breed string `json:"breed"`
}

var registeredDogs = make(map[string]Dog)

func RegisterDog(w http.ResponseWriter, r *http.Request) {
	var dog Dog
	err := json.NewDecoder(r.Body).Decode(&dog)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	registeredDogs[dog.ID] = dog
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(dog)
}

func GetRegisteredDogs(w http.ResponseWriter, r *http.Request) {
	dogs := make([]Dog, 0, len(registeredDogs))
	for _, dog := range registeredDogs {
		dogs = append(dogs, dog)
	}
	json.NewEncoder(w).Encode(dogs)
}

func GetDogByID(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	dogID := vars["id"]

	dog, exists := registeredDogs[dogID]
	if !exists {
		http.Error(w, "Dog not found", http.StatusNotFound)
		return
	}

	json.NewEncoder(w).Encode(dog)
}

func SetupRoutes(r *mux.Router) {
	r.HandleFunc("/register", RegisterDog).Methods("POST")
	r.HandleFunc("/dogs", GetRegisteredDogs).Methods("GET")
	r.HandleFunc("/dogs/{id}", GetDogByID).Methods("GET")
}
