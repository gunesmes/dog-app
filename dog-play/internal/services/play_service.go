package services

import (
    "errors"
    "sync"
)

// Dog represents a registered dog
type Dog struct {
    ID   string
    Name string
}

// PlayService manages play actions for registered dogs
type PlayService struct {
    registeredDogs map[string]Dog
    mu             sync.RWMutex
}

// NewPlayService creates a new PlayService
func NewPlayService() *PlayService {
    return &PlayService{
        registeredDogs: make(map[string]Dog),
    }
}

// RegisterDog registers a new dog
func (ps *PlayService) RegisterDog(dog Dog) {
    ps.mu.Lock()
    defer ps.mu.Unlock()
    ps.registeredDogs[dog.ID] = dog
}

// IsDogRegistered checks if a dog is registered
func (ps *PlayService) IsDogRegistered(dogID string) bool {
    ps.mu.RLock()
    defer ps.mu.RUnlock()
    _, exists := ps.registeredDogs[dogID]
    return exists
}

// Play allows a registered dog to play
func (ps *PlayService) Play(dogID string) (string, error) {
    if !ps.IsDogRegistered(dogID) {
        return "", errors.New("dog not registered")
    }
    return "Dog is playing!", nil
}