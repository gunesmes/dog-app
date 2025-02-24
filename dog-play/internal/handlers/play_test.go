package handlers

import (
	"bytes"
	"encoding/json"
	"github.com/gorilla/mux"
	"github.com/stretchr/testify/assert"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"testing"
)

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

type mockTransport struct {
	response *http.Response
}

func (m *mockTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	return m.response, nil
}
