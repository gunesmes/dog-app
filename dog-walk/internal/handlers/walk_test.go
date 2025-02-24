package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"bytes"
	"io/ioutil"

	"github.com/gorilla/mux"
	"github.com/stretchr/testify/assert"
)

type mockTransport struct {
	response *http.Response
}

func (m *mockTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	return m.response, nil
}

func TestCanWalk(t *testing.T) {
	router := mux.NewRouter()
	router.HandleFunc("/walk/{id}", CanWalk).Methods("GET")

	// Mock the dog-registration service response
	dog := Dog{ID: "1", Name: "Buddy", Breed: "Golden Retriever"}
	dogJSON, _ := json.Marshal(dog)
	http.DefaultClient = &http.Client{
		Transport: &mockTransport{
			response: &http.Response{
				StatusCode: http.StatusOK,
				Body:       ioutil.NopCloser(bytes.NewBuffer(dogJSON)),
			},
		},
	}

	req, _ := http.NewRequest("GET", "/walk/1", nil)
	rr := httptest.NewRecorder()
	router.ServeHTTP(rr, req)

	assert.Equal(t, http.StatusOK, rr.Code)
	var response map[string]string
	json.Unmarshal(rr.Body.Bytes(), &response)
	assert.Equal(t, "This dog can walk", response["message"])
}

func TestCanNotWalk(t *testing.T) {
	router := mux.NewRouter()
	router.HandleFunc("/walk/{id}", CanWalk).Methods("GET")

	// Mock the dog-registration service response
	dog := Dog{ID: "1", Name: "Buddy", Breed: "Golden Retriever"}
	dogJSON, _ := json.Marshal(dog)
	http.DefaultClient = &http.Client{
		Transport: &mockTransport{
			response: &http.Response{
				StatusCode: http.StatusForbidden,
				Body:       ioutil.NopCloser(bytes.NewBuffer(dogJSON)),
			},
		},
	}

	req, _ := http.NewRequest("GET", "/walk/2", nil)
	rr := httptest.NewRecorder()
	router.ServeHTTP(rr, req)

	assert.Equal(t, http.StatusForbidden, rr.Code)
	var response map[string]string
	json.Unmarshal(rr.Body.Bytes(), &response)
	assert.Equal(t, "This dog cannot walk, it should be registered first", response["message"])
}

func TestWalkHasInternalError(t *testing.T) {
	// create later
}

func TestWalkHasParsingError(t *testing.T) {
	// create later
}
