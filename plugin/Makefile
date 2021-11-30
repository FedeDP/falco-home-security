SHELL=/bin/bash -o pipefail

GO ?= go

all: libhomesecurity.so main

main: plugin.go main.go
	$(GO) build .

clean:
	rm -f *.so *.h

libhomesecurity.so: plugin.go main.go
	GODEBUG=cgocheck=2 $(GO) build -buildmode=c-shared -o libhomesecurity.so *.go

