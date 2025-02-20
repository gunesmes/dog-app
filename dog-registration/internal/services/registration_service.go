package services

import (
    "errors"
    "sync"
)

type Dog struct {
    ID   string
    Name string
    Age  int
}

type RegistrationService struct {
    registeredDogs map[string]Dog
    mu             sync.Mutex
}

func NewRegistrationService() *RegistrationService {
    return &RegistrationService{
        registeredDogs: make(map[string]Dog),
    }
}

func (rs *RegistrationService) RegisterDog(dog Dog) error {
    rs.mu.Lock()
    defer rs.mu.Unlock()

    if _, exists := rs.registeredDogs[dog.ID]; exists {
        return errors.New("dog is already registered")
    }

    rs.registeredDogs[dog.ID] = dog
    return nil
}

func (rs *RegistrationService) GetDog(id string) (Dog, error) {
    rs.mu.Lock()
    defer rs.mu.Unlock()

    dog, exists := rs.registeredDogs[id]
    if !exists {
        return Dog{}, errors.New("dog not found")
    }

    return dog, nil
}

func (rs *RegistrationService) IsDogRegistered(id string) bool {
    rs.mu.Lock()
    defer rs.mu.Unlock()

    _, exists := rs.registeredDogs[id]
    return exists
}