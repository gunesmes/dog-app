package services

import (
	"errors"
	"sync"
)

// SleepService manages sleep data for registered dogs.
type SleepService struct {
	dogSleepData map[string]int // map of dog ID to sleep hours
	mu           sync.Mutex
}

// NewSleepService creates a new instance of SleepService.
func NewSleepService() *SleepService {
	return &SleepService{
		dogSleepData: make(map[string]int),
	}
}

// RecordSleep records the sleep hours for a registered dog.
func (s *SleepService) RecordSleep(dogID string, hours int) error {
	if hours < 0 {
		return errors.New("sleep hours cannot be negative")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	s.dogSleepData[dogID] += hours
	return nil
}

// GetSleepHours retrieves the total sleep hours for a registered dog.
func (s *SleepService) GetSleepHours(dogID string) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	hours, exists := s.dogSleepData[dogID]
	if !exists {
		return 0, errors.New("dog not found")
	}
	return hours, nil
}