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

	y := F.Softmax(1)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[2 3]([0.09003057317038046 0.24472847105479764 0.6652409557748218 0.017668422014048047 0.017668422014048047 0.9646631559719038])
	// variable[2 3]([1.3877787807814457e-17 2.7755575615628914e-17 1.1102230246251565e-16 3.469446951953614e-18 3.469446951953614e-18 1.1102230246251565e-16])
}

func ExampleSoftmax_axis0() {
	x := variable.New(
		1, 2, 3,
		4, 4, 8,
	).Reshape(2, 3)

	y := F.Softmax(0)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[2 3]([0.04742587317756679 0.11920292202211755 0.006692850924284856 0.9525741268224334 0.8807970779778823 0.9933071490757153])
	// variable[2 3]([-1.3877787807814457e-17 1.3877787807814457e-17 -1.734723475976807e-18 -2.220446049250313e-16 1.1102230246251565e-16 -2.220446049250313e-16])
}
