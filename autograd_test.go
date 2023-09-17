package autograd_test

import (
	"fmt"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/numerical"
	"github.com/itsubaki/autograd/variable"
)

func Example() {
	x := variable.New(0.5)
	y := F.Square(F.Exp(F.Square(x)...)...)

	y[0].Backward()
	fmt.Println(x.Grad)

	// Output:
	// [3.297442541400256]
}

func Example_add() {
	x := variable.New(2.0)
	y := variable.New(3.0)
	z := F.Add(F.Square(x)[0], F.Square(y)[0])

	z[0].Backward()
	fmt.Println(z)
	fmt.Println(x.Grad)
	fmt.Println(y.Grad)

	// Output:
	// [variable([13])]
	// [4]
	// [6]
}

func Example_func() {
	// p44
	x := variable.New(0.5)
	a := F.Square(x)
	b := F.Exp(a...)
	y := F.Square(b...)

	y[0].Backward()
	fmt.Println(x.Grad)

	// Output:
	// [3.297442541400256]
}

func Example_numericalDiff() {
	// p23
	v := []*variable.Variable{variable.New(0.5)}
	f := func(x ...*variable.Variable) []*variable.Variable {
		A := F.Square
		B := F.Exp
		C := F.Square
		return C(B(A(x...)...)...)
	}

	fmt.Println(numerical.Diff(f, v))

	// Output:
	// [variable([3.2974426293330694])]
}

func Example_creator() {
	x := variable.New(0.5)

	a := F.Square(x)
	b := F.Exp(a...)
	y := F.Square(b...)

	y[0].Backward()

	fmt.Println(x)
	fmt.Println(y)
	fmt.Println(x.Grad)
	fmt.Println()

	// p40
	fmt.Println(y[0].Creator)
	fmt.Println(y[0].Creator.Input()[0] == b[0])
	fmt.Println(y[0].Creator.Input()[0].Creator)
	fmt.Println(y[0].Creator.Input()[0].Creator.Input()[0] == a[0])
	fmt.Println(y[0].Creator.Input()[0].Creator.Input()[0].Creator)
	fmt.Println(y[0].Creator.Input()[0].Creator.Input()[0].Creator.Input()[0] == x)

	// Output:
	// variable([0.5])
	// [variable([1.648721270700128])]
	// [3.297442541400256]
	//
	// *function.SquareT([[1.2840254166877414]])
	// true
	// *function.ExpT([[0.25]])
	// true
	// *function.SquareT([[0.5]])
	// true
}
