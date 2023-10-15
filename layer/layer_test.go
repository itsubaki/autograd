package layer_test

import (
	"fmt"

	L "github.com/itsubaki/autograd/layer"
)

func ExampleLayer() {
	l := &L.Layer{}

	l.Add(L.Linear(1))
	l.Add(L.Linear(2))
	l.Add(L.Linear(3))

	for _, v := range l.Params() {
		fmt.Println(v)
	}

	// Unordered output:
	// b([0])
	// b([0 0])
	// b([0 0 0])
}

func ExampleLayer_linear() {
	l := &L.Layer{
		Forwarder: L.Linear(1),
	}

	for _, v := range l.Params() {
		fmt.Println(v)
	}

	// Output:
	// b([0])
}
