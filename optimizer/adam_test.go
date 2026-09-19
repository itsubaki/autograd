package optimizer_test

import (
	"fmt"

	"github.com/itsubaki/autograd/optimizer"
	"github.com/itsubaki/autograd/variable"
)

func ExampleAdam() {
	p := variable.New(1.0)
	p.Grad = variable.New(1.0)
	m := &TestModel{P: p}

	o := optimizer.Adam{
		Alpha: 0.001,
		Beta1: 0.9,
		Beta2: 0.999,
	}

	for range 2 {
		o.Update(m.Params())
		fmt.Println(p)

	}

	// Output:
	// variable(0.999)
	// variable(0.998)
}

func ExampleAdam_nograd() {
	p := variable.New(1.0)
	m := &TestModel{P: p}

	o := optimizer.Adam{
		Alpha: 0.001,
		Beta1: 0.9,
		Beta2: 0.999,
	}

	o.Update(m.Params())
	fmt.Println(p)

	// Output:
	// variable(1)
}
