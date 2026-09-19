package model_test

import (
	"fmt"

	"github.com/itsubaki/autograd/model"
	"github.com/itsubaki/autograd/rand"
	"github.com/itsubaki/autograd/variable"
)

func ExampleLSTM() {
	m := model.NewLSTM(2, 3)

	for _, name := range m.Layers {
		fmt.Printf("%s %T\n", name, m.L[name])
	}

	// Output:
	// lstm *layer.LSTMT
	// linear *layer.LinearT
}

func ExampleLSTM_backward() {
	m := model.NewLSTM(1, 1, model.WithLSTMSource(rand.Const()))

	x := variable.New(
		1, 2,
	).Reshape(1, 2)

	y := m.Forward(x)
	y.Backward()

	y = m.Forward(x)
	y.Backward()

	for _, name := range m.Layers {
		for k, v := range m.L[name].Params().Seq2() {
			fmt.Printf("%s.%s %v\n", name, k, v.Grad)
		}
	}
}

func ExampleLSTM_ResetState() {
	m := model.NewLSTM(1, 1)

	x := variable.New(
		1, 2,
	).Reshape(1, 2)

	m.Forward(x)
	m.ResetState()
	m.Forward(x)

	for k, v := range m.Params().Seq2() {
		fmt.Println(k, v.Grad)
	}

	// Output:
	// linear.b <nil>
	// linear.w <nil>
	// lstm.h2f.w <nil>
	// lstm.h2i.w <nil>
	// lstm.h2o.w <nil>
	// lstm.h2u.w <nil>
	// lstm.x2f.b <nil>
	// lstm.x2f.w <nil>
	// lstm.x2i.b <nil>
	// lstm.x2i.w <nil>
	// lstm.x2o.b <nil>
	// lstm.x2o.w <nil>
	// lstm.x2u.b <nil>
	// lstm.x2u.w <nil>
}

func ExampleLSTM_Params() {
	m := model.NewLSTM(100, 1)

	x := variable.New(
		1, 2, 3,
	).Reshape(1, 3)

	m.Forward(x)
	for k, v := range m.Params().Seq2() {
		fmt.Println(k, v.Shape())
	}

	// Output:
	// linear.b [1 1]
	// linear.w [100 1]
	// lstm.h2f.w [100 100]
	// lstm.h2i.w [100 100]
	// lstm.h2o.w [100 100]
	// lstm.h2u.w [100 100]
	// lstm.x2f.b [1 100]
	// lstm.x2f.w [3 100]
	// lstm.x2i.b [1 100]
	// lstm.x2i.w [3 100]
	// lstm.x2o.b [1 100]
	// lstm.x2o.w [3 100]
	// lstm.x2u.b [1 100]
	// lstm.x2u.w [3 100]
}

func ExampleLSTM_batch() {
	m := model.NewLSTM(5, 1, model.WithLSTMSource(rand.Const()))

	x := variable.New(
		1, 2,
		3, 4,

		5, 6,
		7, 8,
	).Reshape(2, 2, 2)

	y := m.Forward(x)
	y.Backward()
	m.Cleargrads()

	fmt.Println(y.Shape())
	fmt.Println(x.Grad.Shape())

	for k, v := range m.Params().Seq2() {
		fmt.Println(k, v.Shape())
	}

	// Output:
	// [2 2 1]
	// [2 2 2]
	// linear.b [1 1]
	// linear.w [5 1]
	// lstm.h2f.w [5 5]
	// lstm.h2i.w [5 5]
	// lstm.h2o.w [5 5]
	// lstm.h2u.w [5 5]
	// lstm.x2f.b [1 5]
	// lstm.x2f.w [2 5]
	// lstm.x2i.b [1 5]
	// lstm.x2i.w [2 5]
	// lstm.x2o.b [1 5]
	// lstm.x2o.w [2 5]
	// lstm.x2u.b [1 5]
	// lstm.x2u.w [2 5]
}
