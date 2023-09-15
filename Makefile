SHELL := /bin/bash

test:
	go test -cover $(shell go list ./... | grep -v /vendor/ | grep -v /build/) -v -coverprofile=coverage.txt -covermode=atomic
	go tool cover -html=coverage.txt -o coverage.html
