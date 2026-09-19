package model_test

import (
	"fmt"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/model"
	"github.com/itsubaki/autograd/rand"
	"github.com/itsubaki/autograd/variable"
)

func ExampleMLP() {
	m := model.NewMLP([]int{1, 2, 3})

	for _, name := range m.Layers {
		fmt.Printf("%s %T\n", name, m.L[name])
	}

	// Output:
	// linear[0] *layer.LinearT
	// linear[1] *layer.LinearT
	// linear[2] *layer.LinearT
}

func ExampleMLP_backward() {
	m := model.NewMLP([]int{5, 1},
		model.WithMLPSource(rand.Const()),
		model.WithMLPActivation(F.ReLU),
	)

	x := variable.New(
		1, 2,
	).Reshape(1, 2)

	y := m.Forward(x)
	y.Backward()

	for k, v := range m.Params().Seq2() {
		fmt.Printf("%s %v %.6f\n", k, v.Grad.Shape(), v.Grad.Data.Data)
	}
}

func ExampleMLP_cleargrads() {
	m := model.NewMLP([]int{5, 1},
		model.WithMLPSource(rand.Const()),
		model.WithMLPActivation(F.ReLU),
	)

	x := variable.New(
		1, 2,
	).Reshape(1, 2)

	y := m.Forward(x)
	y.Backward()
	m.Cleargrads()

	for k, v := range m.Params().Seq2() {
		fmt.Println(k, v.Grad)
	}

	// Output:
	// linear[0].b <nil>
	// linear[0].w <nil>
	// linear[1].b <nil>
	// linear[1].w <nil>
}

func ExampleMLP_Params() {
	m := model.NewMLP([]int{5, 1})

	x := variable.New(
		1, 2,
	).Reshape(1, 2)

	m.Forward(x) // gen w
	for k, v := range m.Params().Seq2() {
		fmt.Println(k, v.Shape())
	}

	// Output:
	// linear[0].b [1 5]
	// linear[0].w [2 5]
	// linear[1].b [1 1]
	// linear[1].w [5 1]
}

func ExampleMLP_batch() {
	m := model.NewMLP([]int{5, 1},
		model.WithMLPSource(rand.Const()),
		model.WithMLPActivation(F.ReLU),
	)

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
	// linear[0].b [1 5]
	// linear[0].w [2 5]
	// linear[1].b [1 1]
	// linear[1].w [5 1]
}
