package function_test

import (
	"fmt"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

func ExampleFunction() {
	f := &F.Function{
		Forwarder: &F.SquareT{},
	}

	v := variable.New(1, 2, 3, 4, 5)
	f.Apply(v)

	fmt.Println(f.Input())
	fmt.Println(f.Output())

	// Output:
	// variable([1 2 3 4 5])
	// variable([1 4 9 16 25])
}

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
