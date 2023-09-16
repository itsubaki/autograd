package function_test

import (
	"fmt"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

func ExampleNumericalDiff() {
	// p22
	v := variable.New(2.0)
	f := F.Square

	fmt.Println(F.NumericalDiff(f, v))

	// Output:
	// variable([4.000000000004])
}
