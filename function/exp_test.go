package function_test

import (
	"fmt"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

func ExampleExp() {
	// p16
	A := &F.SquareT{}
	B := &F.ExpT{}
	C := &F.SquareT{}

	x := variable.New(0.5)
	a := A.Apply(x)
	b := B.Apply(a)
	y := C.Apply(b)

	fmt.Println(y)

	b.Grad = C.Backward([]float64{1})
	a.Grad = B.Backward(b.Grad)
	x.Grad = A.Backward(a.Grad)

	fmt.Println(x.Grad)

	// p40
	fmt.Println(y.Creator.Input() == b)
	fmt.Println(y.Creator.Input().Creator == B)
	fmt.Println(y.Creator.Input().Creator.Input() == a)
	fmt.Println(y.Creator.Input().Creator.Input().Creator == A)
	fmt.Println(y.Creator.Input().Creator.Input().Creator.Input() == x)

	// Output:
	// variable([1.648721270700128])
	// [3.297442541400256]
	// true
	// true
	// true
	// true
	// true

}
