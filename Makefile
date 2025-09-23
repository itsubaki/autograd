SHELL := /bin/bash

test:
	go test -cover $(shell go list ./... | grep -v /cmd/ ) -v -coverprofile=coverage.txt -covermode=atomic
	go tool cover -html=coverage.txt -o coverage.html
	golangci-lint run

bench:
	go test -bench . ./... --benchmem

install:
	brew install graphviz
	go install github.com/itsubaki/plot@latest

.PHONY: dot
dot:
	go run cmd/dot/main.go -func tanh -order 4 -verbose > sample.dot; dot sample.dot -T png -o sample.png

lstm:
	go run cmd/lstm/main.go > cos.csv; plot cos.csv
