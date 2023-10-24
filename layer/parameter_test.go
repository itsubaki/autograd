package layer_test

import (
	"fmt"

	"github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/variable"
)

func ExampleParameters_Params() {
	p := make(layer.Parameters)
	p.Add("w", variable.New(1, 2))
	p.Add("b", variable.New(3, 4))

	for _, v := range p.Params() {
		fmt.Println(v)
	}

	// Unordered output:
	// w([1 2])
	// b([3 4])
}

func ExampleParameters_Cleargrads() {
	v := variable.New(1, 2)
	v.Grad = variable.New(3, 4)

	p := make(layer.Parameters)
	p.Add("w", v)
	p.Cleargrads()

	for _, v := range p {
		fmt.Println(v, v.Grad)
	}

	// Output:
	// w([1 2]) <nil>
}
