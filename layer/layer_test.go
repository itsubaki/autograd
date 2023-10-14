package layer_test

import (
	"fmt"

	L "github.com/itsubaki/autograd/layer"
)

func ExampleLayer() {
	l := &L.Layer{
		Forwarder: L.Linear(1),
		Layers: []L.Layer{
			{
				Forwarder: L.Linear(2),
			},
			{
				Forwarder: L.Linear(3),
			},
		},
	}

	for _, v := range l.Params() {
		fmt.Println(v)
	}

	l.Cleargrads()

	// Unordered output:
	// b([0])
	// b([0 0])
	// b([0 0 0])
}
