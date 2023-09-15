package autograd_test

import (
	"fmt"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

func ExampleVariable() {
	x := variable.New(0.5)
	y := F.Square(F.Exp(F.Square(x)))

	y.Backward()
	fmt.Println(x.Grad)

	// Output:
	// [3.297442541400256]
}

func ExampleVariable_func() {
	// p44
	x := variable.New(0.5)
	a := F.Square(x)
	b := F.Exp(a)
	y := F.Square(b)

	y.Backward()
	fmt.Println(x.Grad)

	// Output:
	// [3.297442541400256]
}

func ExampleVariable_type() {
	// p44
	A := &F.SquareT{}
	B := &F.ExpT{}
	C := &F.SquareT{}

	x := variable.New(0.5)
	a := A.Apply(x)
	b := B.Apply(a)
	y := C.Apply(b)

	y.Backward()
	fmt.Println(x.Grad)

	// Output:
	// [3.297442541400256]
}
