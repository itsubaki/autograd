package optimizer_test

import (
	"fmt"

	"github.com/itsubaki/autograd/hook"
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

	o.Update(m)
	fmt.Println(p)

	o.Update(m)
	fmt.Println(p)

	// Output:
	// variable([0.9990000003162277])
	// variable([0.9980000005398904])
}

func ExampleAdam_hook() {
	p := variable.New(1.0)
	p.Grad = variable.New(1.0)
	m := &TestModel{P: p}

	o := optimizer.Adam{
		Alpha: 0.001,
		Beta1: 0.9,
		Beta2: 0.999,
		Hook: []optimizer.Hook{
			hook.WeightDecay(0.1),
		},
	}

	o.Update(m)
	fmt.Println(p)

	o.Update(m)
	fmt.Println(p)

	// Output:
	// variable([1.0990000003162277])
	// variable([1.1980000005398905])
}
