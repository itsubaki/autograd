package model_test

import (
	"fmt"
	"math/rand"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/model"
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
	m := model.NewMLP([]int{5, 1}, model.MLPOpts{
		Activation: F.ReLU,
		Source:     rand.NewSource(1),
	})

	x := variable.New(1, 2)
	y := m.Forward(x)
	y.Backward()

	for _, p := range m.Params() {
		fmt.Println(p.Name, p.Grad)
	}

	// Unordered output:
	// b variable([0 0.3748570762853248 0.5808592854004844 0.2358418430773808 0.32755798713394974])
	// w variable([[0 0.3748570762853248 0.5808592854004844 0.2358418430773808 0.32755798713394974] [0 0.7497141525706496 1.1617185708009687 0.4716836861547616 0.6551159742678995]])
	// b variable([1])
	// w variable([[0] [0.1352468783636501] [1.0305442093147756] [0.582057128601811] [1.198946798274449]])
}

func ExampleMLP_cleargrads() {
	m := model.NewMLP([]int{5, 1}, model.MLPOpts{
		Activation: F.ReLU,
		Source:     rand.NewSource(1),
	})

	x := variable.New(1, 2)
	y := m.Forward(x)
	y.Backward()
	m.Cleargrads()

	for _, p := range m.Params() {
		fmt.Println(p.Name, p.Grad)
	}

	// Unordered output:
	// b <nil>
	// w <nil>
	// b <nil>
	// w <nil>
}

func ExampleMLP_flattenparams() {
	m := model.NewMLP([]int{5, 1})

	x := variable.New(1, 2)
	m.Forward(x) // gen w

	for k, p := range m.FlattenParams() {
		fmt.Println(k, variable.Shape(p))
	}

	// Unordered output:
	// 0.w [2 5]
	// 0.b [1 5]
	// 1.w [5 1]
	// 1.b [1 1]
}
