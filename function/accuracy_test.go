package function_test

import (
	"fmt"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

func ExampleAccuracy() {
	// p391
	y := variable.New(
		0.2, 0.8, 0.0,
		0.1, 0.9, 0.0,
		0.8, 0.1, 0.1,
	).Reshape(3, 3)

	t := variable.New(1, 2, 0)
	fmt.Println(F.Accuracy(y, t))

	// Output:
	// variable(0.6666666666666666)
}
