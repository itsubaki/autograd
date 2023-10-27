package optimizer_test

import (
	"fmt"

	"github.com/itsubaki/autograd/hook"
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

	o.Update(m)
	fmt.Println(p)

	o.Update(m)
	fmt.Println(p)

	// Output:
	// variable([0.999])
	// variable([0.9971])
}

func ExampleMomentum_hook() {
	p := variable.New(1.0)
	p.Grad = variable.New(1.0)
	m := &TestModel{P: p}

	o := optimizer.Momentum{
		LearningRate: 0.001,
		Momentum:     0.9,
		Hooks: []optimizer.Hook{
			hook.WeightDecay(0.1),
		},
	}

	o.Update(m)
	fmt.Println(p)

	o.Update(m)
	fmt.Println(p)

	// Output:
	// variable([1.0990000000000002])
	// variable([1.1971000000000003])
}
