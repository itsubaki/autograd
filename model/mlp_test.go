package model_test

import (
	"fmt"
	"math/rand"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/model"
	"github.com/itsubaki/autograd/variable"
)

func ExampleMLP() {
	s := rand.NewSource(1)
	mlp := model.NewMLP([]int{5, 1}, F.ReLU, s)

	x := variable.New(1, 2)
	y := mlp.Forward(x)
	fmt.Println(y)

	// Output:
	// variable([1.179297448305554])
}

func ExampleMLP_backward() {
	s := rand.NewSource(1)
	mlp := model.NewMLP([]int{5, 1}, F.ReLU, s)

	x := variable.New(1, 2)
	y := mlp.Forward(x)

	y.Backward()
	for _, l := range mlp.Layers {
		for _, p := range l.Params() {
			fmt.Println(p.Name, p.Grad)
		}
	}

	// Unordered output:
	// b variable([0 0.3748570762853248 0.5808592854004844 0.2358418430773808 0.32755798713394974])
	// w variable([[0 0.3748570762853248 0.5808592854004844 0.2358418430773808 0.32755798713394974] [0 0.7497141525706496 1.1617185708009687 0.4716836861547616 0.6551159742678995]])
	// b variable([1])
	// w variable([[0] [0.1352468783636501] [1.0305442093147756] [0.582057128601811] [1.198946798274449]])
}
