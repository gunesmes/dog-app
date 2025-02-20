package services

import (
    "errors"
    "sync"
)

type Dog struct {
    ID   string
    Name string
}

type WalkService struct {
    registeredDogs map[string]Dog
    mu             sync.Mutex
}

func NewWalkService() *WalkService {
    return &WalkService{
        registeredDogs: make(map[string]Dog),
    }
}

func (ws *WalkService) RegisterDog(dog Dog) {
    ws.mu.Lock()
    defer ws.mu.Unlock()
    ws.registeredDogs[dog.ID] = dog
}

func (ws *WalkService) IsDogRegistered(dogID string) (bool, error) {
    ws.mu.Lock()
    defer ws.mu.Unlock()
    if _, exists := ws.registeredDogs[dogID]; !exists {
        return false, errors.New("dog not registered")
    }
    return true, nil
}

func (ws *WalkService) GetRegisteredDogs() []Dog {
    ws.mu.Lock()
    defer ws.mu.Unlock()
    dogs := make([]Dog, 0, len(ws.registeredDogs))
    for _, dog := range ws.registeredDogs {
        dogs = append(dogs, dog)
    }
    return dogs
}