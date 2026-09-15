package optimizer_test

import (
	"fmt"

	"github.com/itsubaki/autograd/optimizer"
	"github.com/itsubaki/autograd/variable"
)

func ExampleMomentum() {
	p := variable.New(1.0)
	p.Grad = variable.New(1.0)
	m := &TestModel{P: p}

	o := optimizer.Momentum{
		LearningRate: 0.001,
		Momentum:     0.9,
	}

	for range 2 {
		o.Update(m.Params())
		fmt.Println(p)
	}

	// Output:
	// variable(0.999)
	// variable(0.9971)
}

func ExampleMomentum_nograd() {
	p := variable.New(1.0)
	m := &TestModel{P: p}

	o := optimizer.Momentum{
		LearningRate: 0.001,
		Momentum:     0.9,
	}

	o.Update(m.Params())
	fmt.Println(p)

	// Output:
	// variable(1)
}
