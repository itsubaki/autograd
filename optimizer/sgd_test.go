package optimizer_test

import (
	"fmt"

	"github.com/itsubaki/autograd/hook"
	"github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/optimizer"
	"github.com/itsubaki/autograd/variable"
)

type TestModel struct {
	P layer.Parameter
}

func (m *TestModel) Params() []layer.Parameter {
	return []layer.Parameter{m.P}
}

func ExampleSGD() {
	p := variable.New(1.0)
	p.Grad = variable.New(1.0)
	m := &TestModel{P: p}

	o := optimizer.SGD{
		LearningRate: 0.1,
	}
	o.Update(m)

	fmt.Println(p)

	// Output:
	// variable([0.9])
}

func ExampleSGD_hook() {
	p := variable.New(1.0)
	p.Grad = variable.New(1.0)
	m := &TestModel{P: p}

	o := optimizer.SGD{
		LearningRate: 0.1,
		Hooks: []optimizer.Hook{
			hook.WeightDecay(0.1),
		},
	}
	o.Update(m)

	fmt.Println(p)

	// Output:
	// variable([1])
}
