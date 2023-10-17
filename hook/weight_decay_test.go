package hook_test

import (
	"fmt"

	"github.com/itsubaki/autograd/hook"
	"github.com/itsubaki/autograd/variable"
)

func ExampleWeightDecay() {
	p := variable.New(1.0)
	p.Grad = variable.New(1.0)

	h := hook.WeightDecay(0.1)
	h([]*variable.Variable{p})

	fmt.Println(p)

	// Output:
	// variable([1.1])
}
