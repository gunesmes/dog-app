package handlers

import (
	"bytes"
	"encoding/json"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gorilla/mux"
	"github.com/stretchr/testify/assert"
)

type mockTransport struct {
	response *http.Response
}

func (m *mockTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	return m.response, nil
}

func TestCanPlay(t *testing.T) {
	router := mux.NewRouter()
	router.HandleFunc("/play/{id}", CanPlay).Methods("GET")

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

	req, _ := http.NewRequest("GET", "/play/1", nil)
	rr := httptest.NewRecorder()
	router.ServeHTTP(rr, req)

	assert.Equal(t, http.StatusOK, rr.Code)
	var response map[string]string
	json.Unmarshal(rr.Body.Bytes(), &response)
	assert.Equal(t, "This dog can play", response["message"])
}

func TestCanNotPlay(t *testing.T) {
	router := mux.NewRouter()
	router.HandleFunc("/play/{id}", CanPlay).Methods("GET")

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

	req, _ := http.NewRequest("GET", "/play/2", nil)
	rr := httptest.NewRecorder()
	router.ServeHTTP(rr, req)

	assert.Equal(t, http.StatusForbidden, rr.Code)
	var response map[string]string
	json.Unmarshal(rr.Body.Bytes(), &response)
	assert.Equal(t, "This dog cannot play, it should be registered first", response["message"])
}

func TestPlayHasInternalError(t *testing.T) {
	// create later
}

func TestPlayHasParsingError(t *testing.T) {
	// create later
}
