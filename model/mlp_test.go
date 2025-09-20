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

	for _, l := range m.Layers {
		fmt.Printf("%T\n", l)
	}

	// Output:
	// *layer.LinearT
	// *layer.LinearT
	// *layer.LinearT
}

func ExampleMLP_backward() {
	m := model.NewMLP([]int{5, 1},
		model.WithMLPSource(rand.Const()),
		model.WithMLPActivation(F.ReLU),
	)

	x := variable.New(1, 2)
	y := m.Forward(x)
	y.Backward()

	for k, v := range m.Params().Seq2() {
		fmt.Println(k, v.Grad)
	}

	// Output:
	// 0.b variable[1 5]([0 0 -0.11785627150007956 -0.17275376822987032 -0.1452836777854009])
	// 0.w variable[2 5]([[0 0 -0.11785627150007956 -0.17275376822987032 -0.1452836777854009] [0 0 -0.23571254300015912 -0.34550753645974064 -0.2905673555708018]])
	// 1.b variable(1)
	// 1.w variable[5 1]([[0] [0] [1.62887541989766] [0.7662326556923662] [1.9766127473463149]])
}

func ExampleMLP_cleargrads() {
	m := model.NewMLP([]int{5, 1},
		model.WithMLPSource(rand.Const()),
		model.WithMLPActivation(F.ReLU),
	)

	x := variable.New(1, 2)
	y := m.Forward(x)
	y.Backward()
	m.Cleargrads()

	for k, v := range m.Params().Seq2() {
		fmt.Println(k, v.Grad)
	}

	// Output:
	// 0.b <nil>
	// 0.w <nil>
	// 1.b <nil>
	// 1.w <nil>
}

func ExampleMLP_Params() {
	m := model.NewMLP([]int{5, 1})

	x := variable.New(1, 2)
	m.Forward(x) // gen w

	for k, v := range m.Params().Seq2() {
		fmt.Println(k, v.Shape())
	}

	// Output:
	// 0.b [1 5]
	// 0.w [2 5]
	// 1.b [1 1]
	// 1.w [5 1]
}
