SHELL=/bin/bash -o pipefail

GO ?= go

all: libhomesecurity.so

clean:
	rm -f *.so *.h

libhomesecurity.so: main.go # todo(leogr): add other source files
	GODEBUG=cgocheck=2 $(GO) build -buildmode=c-shared -o libhomesecurity.so *.go

