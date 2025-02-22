package main

import (
	"log"
	"net/http"

	"dog-sleep/internal/handlers"
)

func main() {
	http.HandleFunc("/sleep", handlers.NewSleepHandler().HandleSleep)

	log.Println("Starting dog-sleep service on port :8081")
	if err := http.ListenAndServe(":8081", nil); err != nil {
		log.Fatalf("Could not start server: %s\n", err)
	}
}
