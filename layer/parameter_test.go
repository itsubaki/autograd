package layer_test

import (
	"fmt"

	"github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/variable"
)

func ExampleParameter() {
	p := make(layer.Parameters)
	p.Add("w", variable.New(1, 2))
	p.Add("b", variable.New(3, 4))

	for _, v := range p {
		fmt.Println(v)
	}

	// Unordered output:
	// w([1 2])
	// b([3 4])
}
