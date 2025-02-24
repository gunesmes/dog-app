package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gorilla/mux"
	"github.com/stretchr/testify/assert"
)

func TestRegisterDog(t *testing.T) {
	router := mux.NewRouter()
	router.HandleFunc("/register", RegisterDog).Methods("POST")

	dog := Dog{ID: "1", Name: "Buddy", Breed: "Golden Retriever"}
	dogJSON, _ := json.Marshal(dog)

	req, _ := http.NewRequest("POST", "/register", bytes.NewBuffer(dogJSON))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	router.ServeHTTP(rr, req)

	assert.Equal(t, http.StatusCreated, rr.Code)
	var registeredDog Dog
	json.Unmarshal(rr.Body.Bytes(), &registeredDog)
	assert.Equal(t, dog, registeredDog)
}

func TestGetRegisteredDogs(t *testing.T) {
	router := mux.NewRouter()
	router.HandleFunc("/dogs", GetRegisteredDogs).Methods("GET")

	registeredDogs["1"] = Dog{ID: "1", Name: "Buddy", Breed: "Golden Retriever"}

	req, _ := http.NewRequest("GET", "/dogs", nil)
	rr := httptest.NewRecorder()
	router.ServeHTTP(rr, req)

	assert.Equal(t, http.StatusOK, rr.Code)
	var dogs []Dog
	json.Unmarshal(rr.Body.Bytes(), &dogs)
	assert.Equal(t, 1, len(dogs))
	assert.Equal(t, "Buddy", dogs[0].Name)
}

func TestGetDogByID(t *testing.T) {
	router := mux.NewRouter()
	router.HandleFunc("/dogs/{id}", GetDogByID).Methods("GET")

	registeredDogs["1"] = Dog{ID: "1", Name: "Buddy", Breed: "Golden Retriever"}

	req, _ := http.NewRequest("GET", "/dogs/1", nil)
	rr := httptest.NewRecorder()
	router.ServeHTTP(rr, req)

	assert.Equal(t, http.StatusOK, rr.Code)
	var dog Dog
	json.Unmarshal(rr.Body.Bytes(), &dog)
	assert.Equal(t, "Buddy", dog.Name)
}

func TestGetDogByID_NotFound(t *testing.T) {
	router := mux.NewRouter()
	router.HandleFunc("/dogs/{id}", GetDogByID).Methods("GET")

	req, _ := http.NewRequest("GET", "/dogs/2", nil)
	rr := httptest.NewRecorder()
	router.ServeHTTP(rr, req)

	assert.Equal(t, http.StatusNotFound, rr.Code)
}
