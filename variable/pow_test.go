package variable_test

import (
	"fmt"
	"testing"

	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

func ExamplePowT() {
	x := variable.New(3.0)
	f := variable.PowT{P: 4.0}

	fmt.Println(x)
	fmt.Println(f.Forward(x))
	fmt.Println(f.Backward(variable.OnesLike(x)))

	// Output:
	// variable(3)
	// [variable(81)]
	// [variable(108)]
}

func ExamplePow() {
	x := variable.New(2.0)
	y := variable.Pow(3.0)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable(8)
	// variable(12)
}

func ExamplePow_double() {
	x := variable.New(2.0)

	y := variable.Pow(3.0)(x)
	y.Backward(variable.Opts{CreateGraph: true})
	fmt.Println(y)
	fmt.Println(x.Grad)

	for range 3 {
		gx := x.Grad
		x.Cleargrad()
		gx.Backward(variable.Opts{CreateGraph: true})
		fmt.Println(x.Grad)
	}

	// Output:
	// variable(8)
	// variable(12)
	// variable(12)
	// variable(6)
	// variable(0)
}

func TestPow_zeroExponent(t *testing.T) {
	x := variable.New(0, -2, 3, 0).Reshape(2, 2)
	y := variable.Pow(0)(x)
	y.Grad = variable.New(1, 2, 3, 4).Reshape(2, 2)
	y.Backward()

	if !tensor.IsCloseAll(y.Data, tensor.Ones[float32](2, 2)) {
		t.Errorf("y=%v, want all ones", y)
	}

	if x.Grad == nil {
		t.Fatal("missing input gradient")
	}

	if !tensor.SliceEqual(x.Grad.Shape(), x.Shape()) || !tensor.IsCloseAll(x.Grad.Data, tensor.Zeros[float32](2, 2)) {
		t.Errorf("gx=%v, want zeros with shape %v", x.Grad, x.Shape())
	}
}
