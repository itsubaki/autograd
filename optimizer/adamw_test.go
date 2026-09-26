package optimizer_test

import (
	"fmt"

	"github.com/itsubaki/autograd/optimizer"
	"github.com/itsubaki/autograd/variable"
)

func ExampleAdamW() {
	p := variable.New(1.0)
	p.Grad = variable.New(1.0)
	m := &TestModel{P: p}

	o := optimizer.AdamW{
		Alpha:       0.001,
		Beta1:       0.9,
		Beta2:       0.999,
		WeightDecay: 0.1,
	}

	for range 2 {
		o.Update(m.Params())
		fmt.Println(p)
	}

	// Output:
	// variable(0.9989684)
	// variable(0.99794495)
}

func ExampleAdamW_nograd() {
	p := variable.New(1.0)
	m := &TestModel{P: p}

	o := optimizer.AdamW{
		Alpha:       0.001,
		Beta1:       0.9,
		Beta2:       0.999,
		WeightDecay: 0.1,
	}

	o.Update(m.Params())
	fmt.Println(p)

	// Output:
	// variable(1)
}
