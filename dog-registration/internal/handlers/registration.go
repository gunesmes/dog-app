package handlers

import (
	"encoding/json"
	"net/http"
	"strconv"
	"sync"

	"github.com/gorilla/mux"
)

type Dog struct {
	ID    int    `json:"id"`
	Name  string `json:"name"`
	Breed string `json:"breed"`
	Age   int    `json:"age"`
}

var (
	registeredDogs = make(map[int]Dog)
	mu             sync.Mutex
	counter        int
)

func RegisterDog(w http.ResponseWriter, r *http.Request) {
	var dog Dog
	err := json.NewDecoder(r.Body).Decode(&dog)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	mu.Lock()
	defer mu.Unlock()

	// Generate a unique ID for the new dog
	counter++
	dog.ID = counter

	if _, exists := registeredDogs[dog.ID]; exists {
		http.Error(w, "dog is already registered", http.StatusConflict)
		return
	}

	registeredDogs[dog.ID] = dog
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(dog)
}

func GetRegisteredDogs(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	defer mu.Unlock()

	dogs := make([]Dog, 0, len(registeredDogs))
	for _, dog := range registeredDogs {
		dogs = append(dogs, dog)
	}
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(dogs)
}

func GetDogByID(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	dogID, err := strconv.Atoi(vars["id"])
	if err != nil {
		http.Error(w, "invalid dog ID", http.StatusBadRequest)
		return
	}

	mu.Lock()
	defer mu.Unlock()

	dog, exists := registeredDogs[dogID]
	if !exists {
		http.Error(w, "dog not found", http.StatusNotFound)
		return
	}

	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(dog)
}

func DeleteAllDogs(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	defer mu.Unlock()

	// Iterate over the map and delete each entry
	for id := range registeredDogs {
		delete(registeredDogs, id)
	}

	w.WriteHeader(http.StatusOK)
	w.Write([]byte("All dogs deleted"))
}

// routers setup
func SetupRoutes(r *mux.Router) {
	r.HandleFunc("/register", RegisterDog).Methods("POST")
	r.HandleFunc("/dogs", GetRegisteredDogs).Methods("GET")
	r.HandleFunc("/dogs/{id}", GetDogByID).Methods("GET")
	r.HandleFunc("/dogs", DeleteAllDogs).Methods("DELETE")
}
