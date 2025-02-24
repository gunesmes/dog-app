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

func CanWalk(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	dogID := vars["id"]

	// Check if the dog is registered
	resp, err := http.Get(fmt.Sprintf("http://dog-registration:8084/dogs/%s", dogID))
	if err != nil || resp.StatusCode != http.StatusOK {
		w.WriteHeader(http.StatusForbidden)
		json.NewEncoder(w).Encode(map[string]string{"message": "This dog cannot walk, it should be registered first"})
		return
	}

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"message": "Error reading response from registration service"})
		return
	}

	var dog Dog
	err = json.Unmarshal(body, &dog)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"message": "Error parsing response from registration service"})
		return
	}

	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{"message": "This dog can walk"})
}

func SetupRoutes(r *mux.Router) {
	r.HandleFunc("/walk/{id}", CanWalk).Methods("GET")
}
