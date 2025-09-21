package function_test

import (
	"fmt"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

func ExampleSoftmax() {
	x := variable.New(
		1, 2, 3,
		4, 4, 8,
	).Reshape(2, 3)

	y := F.Softmax(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[2 3]([[0.09003057317038046 0.24472847105479764 0.6652409557748218] [0.017668422014048047 0.017668422014048047 0.9646631559719038]])
	// variable[2 3]([[1.3877787807814457e-17 2.7755575615628914e-17 1.1102230246251565e-16] [3.469446951953614e-18 3.469446951953614e-18 1.1102230246251565e-16]])
}
