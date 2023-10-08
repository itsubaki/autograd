package function_test

import (
	"fmt"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

func ExampleSoftmaxSimple() {
	x := variable.NewOf(
		[]float64{1, 2, 3},
		[]float64{4, 4, 8},
	)

	y := F.SoftmaxSimple(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable([[0.09003057317038046 0.24472847105479767 0.6652409557748219] [0.017668422014048047 0.017668422014048047 0.9646631559719039]])
	// variable([[0 0 0] [0 0 0]])
}

func Example_softmax1d() {
	softmax1d := func(x *variable.Variable) *variable.Variable {
		y := F.Exp(x)
		sumy := F.Sum(y)
		return F.Div(y, sumy)
	}

	x := variable.New(0.2, -0.4)
	y := softmax1d(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable([0.6456563062257954 0.3543436937742045])
	// variable([0 0])
}
