SHELL := /bin/bash

test:
	go test -cover $(shell go list ./... | grep -v /vendor/ | grep -v /build/) -v -coverprofile=coverage.txt -covermode=atomic
	go tool cover -html=coverage.txt -o coverage.html

install:
	brew install graphviz

.PHONY: dot
dot:
	go run cmd/main.go -func tanh -order 4 -verbose > sample.dot; dot sample.dot -T png -o sample.png
