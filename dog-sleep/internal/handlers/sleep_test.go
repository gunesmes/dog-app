package handlers

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gorilla/mux"
	"github.com/stretchr/testify/assert"
)

type Sleep struct {
	ID       string `json:"id"`
	Duration int    `json:"duration"`
}

func TestGetSleep(t *testing.T) {
	router := mux.NewRouter()
	router.HandleFunc("/sleep", NewSleepHandler().HandleSleep).Methods("POST")

	req, _ := http.NewRequest("POST", "/sleep", nil)
	rr := httptest.NewRecorder()
	router.ServeHTTP(rr, req)

	assert.Equal(t, http.StatusOK, rr.Code)
	assert.Equal(t, "Dog is sleeping", rr.Body.String())
}

func TestGetSleepByID_NotFound(t *testing.T) {
	router := mux.NewRouter()
	router.HandleFunc("/sleep/{id}", NewSleepHandler().HandleSleep).Methods("POST")

	req, _ := http.NewRequest("GET", "/sleep/1", nil)
	rr := httptest.NewRecorder()
	router.ServeHTTP(rr, req)

	assert.Equal(t, http.StatusMethodNotAllowed, rr.Code)
}
