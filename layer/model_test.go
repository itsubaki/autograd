package layer_test

import (
	"fmt"

	L "github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/variable"
)

func ExampleModel_Cleargrads() {
	m := &L.Model{
		Layers: []*L.Layer{
			L.Linear(5),
		},
	}

	y := m.Layers[0].ApplyAndFirst(variable.New(1, 2, 3))
	y.Backward()

	for _, v := range m.Layers[0].Params() {
		fmt.Println(v.Grad)
	}

	m.Cleargrads()
	for _, v := range m.Layers[0].Params() {
		fmt.Println(v.Grad)
	}

	// Output:
	// variable([1 1 1 1 1])
	// variable([[1 1 1 1 1] [2 2 2 2 2] [3 3 3 3 3]])
	// <nil>
	// <nil>
}
