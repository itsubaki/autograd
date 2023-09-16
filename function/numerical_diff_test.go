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

func ExampleNumericalDiff_chain() {
	// p23
	v := variable.New(0.5)
	f := func(x *variable.Variable) *variable.Variable {
		A := F.Square
		B := F.Exp
		C := F.Square
		return C(B(A(x)))
	}

	fmt.Println(F.NumericalDiff(f, v))

	// Output:
	// variable([3.2974426293330694])
}
